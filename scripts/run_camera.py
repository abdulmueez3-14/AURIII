"""Live webcam runner for the AURI shoulder analysis pipeline.

Privacy: no video frames or images are saved or transmitted. Only numeric per-frame
results are written to a CSV under runs/ (PRIV-REQ). This is a measurement-engine test
harness, NOT a clinical tool: values are uncalibrated geometric estimates.

Usage (from the repo root, inside the venv):
    python scripts/run_camera.py --arm both --torso shoulders --duration 90
Keys (preview window): q = quit, r = recalibrate.

--torso hips        measured spine from both hips (needs hips in view; enables lean/hiking)
--torso shoulders   assumed-vertical frame from the shoulders only (works seated / close up;
                    trunk lean is then confounded with arm motion, lean/hiking disabled)
"""

import argparse
import collections
import csv
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from auri_cv.calibration import ChannelStatus
from auri_cv.movements import MotionStatus, ShoulderMovement
from auri_cv.pipeline import FrameResult, ShoulderAnalysisPipeline, TorsoMode
from auri_cv.pose.data_models import PoseObservation
from auri_cv.pose.landmarker import AuriPoseLandmarker, VisionTaskRunningMode

MIN_VIS = 0.5
KEY_LANDMARKS = [
    f"{side}_{part}" for side in ("LEFT", "RIGHT") for part in ("SHOULDER", "ELBOW", "WRIST", "HIP")
]
M = ShoulderMovement
ABBREV = {
    M.FLEXION: "FLX", M.EXTENSION: "EXT", M.ABDUCTION: "ABD", M.ADDUCTION: "ADD",
    M.SCAPTION: "SCP", M.HORIZONTAL_ADDUCTION: "HAD", M.HORIZONTAL_ABDUCTION: "HAB",
    M.EXTERNAL_ROTATION: "ER", M.INTERNAL_ROTATION: "IR",
}  # fmt: skip
# (title, movement or None, seconds, instruction). Both arms are prompted together.
GUIDED: list[tuple[str, ShoulderMovement | None, float, str]] = [
    ("CALIBRATE", None, 6.0, "Arms hanging relaxed at your sides. Stay still."),
    ("FLEXION", M.FLEXION, 12.0, "Raise BOTH arms straight FORWARD to shoulder height, hold, lower."),
    ("EXTENSION", M.EXTENSION, 10.0, "Arms hanging: swing both arms straight BACK behind you, hold, return."),
    ("ABDUCTION", M.ABDUCTION, 12.0, "Raise BOTH arms straight out to the SIDES to shoulder height, hold, lower."),
    ("ADDUCTION", M.ADDUCTION, 10.0, "Arms hanging: bring one arm ACROSS the front of your body, then the other."),
    ("SCAPTION", M.SCAPTION, 12.0, "Raise both arms diagonally (30 deg in front of the sides, thumbs up) to shoulder height or higher, hold, lower."),
    ("HORIZONTAL ADDUCTION", M.HORIZONTAL_ADDUCTION, 12.0, "Arms out to the sides at shoulder height, then swing both forward across your chest, and back out."),
    ("HORIZONTAL ABDUCTION", M.HORIZONTAL_ABDUCTION, 12.0, "Arms in front at shoulder height, then swing them back and out past your sides, and return."),
    ("EXTERNAL ROTATION", M.EXTERNAL_ROTATION, 12.0, "Elbows bent 90 deg tight at your sides, forearms forward: rotate hands OUTWARD, return."),
    ("INTERNAL ROTATION", M.INTERNAL_ROTATION, 12.0, "Same start: rotate hands INWARD across your belly, return."),
]  # fmt: skip
LEAD_S = 5.0  # "get ready" time before each step; frames in it are never scored
GUIDED_TOTAL_S = sum(g[2] + LEAD_S for g in GUIDED)


def _step_at(t_s: float) -> tuple[int, float, bool] | None:
    """(step index, seconds left in this phase, in_prep) for t seconds after the countdown."""
    acc = 0.0
    for i, (_, _, dur, _) in enumerate(GUIDED):
        if t_s < acc + LEAD_S:
            return i, acc + LEAD_S - t_s, True
        acc += LEAD_S
        if t_s < acc + dur:
            return i, acc + dur - t_s, False
        acc += dur
    return None


def _beep(freq: int) -> None:
    try:  # audible cue on step changes (Windows); silently skipped elsewhere
        import winsound

        winsound.Beep(freq, 160)
    except Exception:  # noqa: BLE001
        pass


BONES = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
]
CSV_FIELDS = [
    "arm", "t_ms", "detected", "quality", "quality_score", "elev_raw", "flex_raw", "abd_raw",
    "plane_raw", "elev_filt", "channel", "relative", "display", "rep_state", "rep_count",
    "lean_lat", "lean_fwd", "trunk_rot", "hiking_m", "warnings", "weak_landmarks", "latency_ms",
    "movement", "motion_angle", "motion_status", "motion_conf", "view", "view_yaw", "suggested",
    "motion_note", "prompt", "prompt_conf",
    *[f"ang_{m.value}" for m in ShoulderMovement],
    *[f"st_{m.value}" for m in ShoulderMovement],
]  # fmt: skip


def _weak(r: FrameResult) -> tuple[str, ...]:
    return tuple(r.quality.low_confidence_landmarks) + tuple(r.quality.missing_landmarks)


def _motion_cols(r: FrameResult) -> dict[str, str]:
    m, sg = r.motion, r.suggestion
    return {
        "movement": m.movement.value if m else "",
        "motion_angle": _fmt(m.angle_deg if m else None),
        "motion_status": m.status.value if m else "",
        "motion_conf": _fmt(m.confidence if m else None),
        "view": m.view.value if m else "",
        "view_yaw": _fmt(m.view_yaw_deg if m else None),
        "suggested": (
            sg.movement.value if sg.movement else "|".join(c.value for c in sg.candidates)
        )
        if sg
        else "",
        "motion_note": m.reasons[0] if m and m.angle_deg is None else "",
    }


def _fmt(v: float | None) -> str:
    return "" if v is None else f"{v:.3f}"


def _row(
    arm: str,
    t_ms: float,
    det: bool,
    r: FrameResult,
    latency_ms: float,
    prompt: str = "",
    prompt_mv: ShoulderMovement | None = None,
) -> dict[str, str]:
    c = r.compensation
    pe = (r.all_motions or {}).get(prompt_mv) if prompt_mv else None
    cols = _motion_cols(r)
    if pe is not None and pe.angle_deg is None:
        cols["motion_note"] = pe.reasons[0]  # why the PROMPTED movement has no value
    return {
        "arm": arm,
        "t_ms": f"{t_ms:.1f}",
        "detected": str(int(det)),
        "quality": r.quality.status.value,
        "quality_score": f"{r.quality.quality_score:.3f}",
        "elev_raw": _fmt(r.raw.elevation_deg if r.raw else None),
        "flex_raw": _fmt(r.raw.flexion_deg if r.raw else None),
        "abd_raw": _fmt(r.raw.abduction_deg if r.raw else None),
        "plane_raw": _fmt(r.raw.plane_of_elevation_deg if r.raw else None),
        "elev_filt": _fmt(r.filtered_elevation_deg),
        "channel": r.channel.status.value,
        "relative": _fmt(r.channel.relative_deg),
        "display": _fmt(r.channel.display_deg),
        "rep_state": r.rep_state.value,
        "rep_count": str(r.rep_count),
        "lean_lat": _fmt(c.lateral_lean_deg if c else None),
        "lean_fwd": _fmt(c.forward_lean_deg if c else None),
        "trunk_rot": _fmt(c.trunk_rotation_deg if c else None),
        "hiking_m": _fmt(c.shoulder_hiking_m if c else None),
        "warnings": "|".join(w.value for w in c.warnings) if c else "",
        "weak_landmarks": "|".join(_weak(r)),
        "latency_ms": f"{latency_ms:.1f}",
        **cols,
        "prompt": prompt,
        "prompt_conf": _fmt(pe.confidence if pe and pe.angle_deg is not None else None),
        **{f"ang_{m.value}": _fmt(e.angle_deg) for m, e in (r.all_motions or {}).items()},
        **{f"st_{m.value}": e.status.value for m, e in (r.all_motions or {}).items()},
    }


def _panel(
    frame: np.ndarray,
    lines: list[str],
    x: int,
    y: int,
    accent: tuple[int, int, int],
    scale: float = 0.55,
    thick: int = 1,
):
    """Text on a solid dark panel so it never blends with the camera image."""
    font, pad = cv2.FONT_HERSHEY_SIMPLEX, 6
    sizes = [cv2.getTextSize(t, font, scale, thick)[0] for t in lines]
    w = max(s[0] for s in sizes) + 2 * pad
    line_h = max(s[1] for s in sizes) + 8
    h = line_h * len(lines) + 2 * pad
    cv2.rectangle(frame, (x, y), (x + w, y + h), (20, 20, 20), -1)
    cv2.rectangle(frame, (x, y), (x + 5, y + h), accent, -1)
    for i, text in enumerate(lines):
        cv2.putText(
            frame, text, (x + pad + 4, y + pad + line_h * (i + 1) - 6), font, scale,
            (255, 255, 255), thick, cv2.LINE_AA,
        )  # fmt: skip
    return h


def _compact(r: FrameResult) -> list[str]:
    """Two lines: every movement's angle ('-' = not measurable now, '~' = low confidence)."""
    if not r.all_motions:
        return []
    parts = []
    for m, e in r.all_motions.items():
        if e.angle_deg is None:
            parts.append(f"{ABBREV[m]} -")
        else:
            mark = "~" if e.status == MotionStatus.LOW_CONFIDENCE else ""
            parts.append(f"{ABBREV[m]} {e.angle_deg:+.0f}{mark}")
    return ["  ".join(parts[:5]), "  ".join(parts[5:])]


def _wrap(text: str, width: int = 62) -> list[str]:
    out, cur = [], ""
    for word in text.split():
        if len(cur) + len(word) + 1 > width:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return [*out, cur]


def _draw(
    frame,
    pose: PoseObservation,
    results: dict[str, FrameResult | None],
    msg: str,
    prompt: list[str] | None = None,
) -> None:
    """Draws on the MIRRORED preview; landmark x is mapped to 1 - x accordingly."""
    h, w = frame.shape[:2]

    def pt(name: str) -> tuple[int, int] | None:
        lm = pose.get_landmark_2d(name)
        return None if lm is None else (int(round((1.0 - lm.x) * w)), int(round(lm.y * h)))

    def vis(name: str) -> float:
        lm = pose.get_landmark_3d(name)
        return 0.0 if lm is None else lm.visibility

    for a, b in BONES:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            ok = min(vis(a), vis(b)) >= MIN_VIS
            cv2.line(frame, pa, pb, (0, 200, 0) if ok else (0, 0, 200), 3 if ok else 1)
    for name in KEY_LANDMARKS:
        p = pt(name)
        if p:
            good = vis(name) >= MIN_VIS
            cv2.circle(frame, p, 7, (0, 255, 0) if good else (0, 0, 255), -1)
            cv2.circle(frame, p, 8, (255, 255, 255), 1)

    y = 8
    y += _panel(frame, [msg], 8, y, (255, 200, 0)) + 6
    if prompt:
        y += _panel(frame, prompt, 8, y, (0, 165, 255), scale=0.9, thick=2) + 6
    for arm, r in results.items():
        if r is None:
            continue
        ch = r.channel
        disp = "---" if ch.display_deg is None else f"{ch.display_deg:+.1f} deg"
        raw = "---" if r.raw is None else f"{r.raw.elevation_deg:.0f} deg"
        lines = [
            f"{arm.upper()} ARM  [{ch.status.value}]",
            f"excursion: {disp}   raw elev: {raw}   reps: {r.rep_count}",
        ]
        if r.motion is not None:
            m = r.motion
            if m.angle_deg is None:
                lines.append(f"{m.movement.value}: NO VALUE - {m.reasons[0]}")
            else:
                lines.append(
                    f"{m.movement.value}: {m.angle_deg:+.1f} deg  conf {m.confidence:.2f} "
                    f"[{m.status.value}]  view {m.view.value}"
                )
            if r.suggestion and r.suggestion.movement and r.suggestion.movement != m.movement:
                lines.append(f"looks like: {r.suggestion.movement.value}")
        lines += _compact(r)
        if _weak(r):
            lines.append("cannot see: " + ", ".join(n.lower().replace("_", " ") for n in _weak(r)))
        if r.compensation and r.compensation.warnings:
            lines.append("WARN: " + " ".join(x.value for x in r.compensation.warnings))
        y += _panel(frame, lines, 8, y, (0, 200, 0) if ch.status == ChannelStatus.OK else (0, 0, 200)) + 6  # fmt: skip


def _coverage(rows: list[dict[str, str]]) -> None:
    """Per prompted movement and arm: how often it was measurable and what values it gave."""
    print("\n===== GUIDED COVERAGE (prompted movement, per arm) =====")
    print(f"{'step':22s} {'arm':5s} {'frames':>6s} {'measurable':>10s} {'peak':>7s} "
          f"{'median conf':>11s}  main reason when not measurable")  # fmt: skip
    for title, mv, _, _ in GUIDED:
        if mv is None:
            continue
        for arm in ("right", "left"):
            rs = [r for r in rows if r["arm"] == arm and r["prompt"] == title]
            if not rs:
                continue
            ang = [float(r[f"ang_{mv.value}"]) for r in rs if r.get(f"ang_{mv.value}")]
            why = collections.Counter(
                r["motion_note"] for r in rs if not r.get(f"ang_{mv.value}") and r["motion_note"]
            )
            confs = [float(r["prompt_conf"]) for r in rs if r["prompt_conf"]]
            mc = f"{statistics.median(confs):.2f}" if confs else "-"
            pct = 100.0 * len(ang) / len(rs)
            peak = f"{round(max(ang)):d}" if ang else "-"
            print(f"{title:22s} {arm:5s} {len(rs):6d} {pct:9.0f}% {peak:>7s} "
                  f"{mc:>11s}  {why.most_common(1)[0][0][:48] if why else ''}")  # fmt: skip


def _summary(rows: list[dict[str, str]], pipes: dict[str, ShoulderAnalysisPipeline], wall_s: float):
    print("\n===== SESSION SUMMARY =====")
    for arm, pipe in pipes.items():
        mine = [r for r in rows if r["arm"] == arm]
        n = len(mine)
        det = sum(int(r["detected"]) for r in mine)
        ok = [r for r in mine if r["channel"] == ChannelStatus.OK.value]
        lat = [float(r["latency_ms"]) for r in mine]
        print(f"--- {arm.upper()} arm ---")
        print(f"frames: {n}  effective fps: {n / max(wall_s, 1e-9):.1f}  detected: {det}/{n}")
        print(f"frames with valid calibrated reading: {len(ok)}")
        if lat:
            print(f"latency ms: median {statistics.median(lat):.1f}, max {max(lat):.1f}")
        b = pipe.channel.baseline
        print("baseline: " + (f"{b.theta0_deg:.1f} deg (IQR {b.iqr_deg:.2f})" if b else "NOT SET"))
        elev = [float(r["elev_raw"]) for r in mine if r["elev_raw"]]
        if elev:
            print(f"raw elevation range: {min(elev):.1f} .. {max(elev):.1f} deg")
        f = pipe.fsm
        print(f"reps: {f.rep_count} (invalidated {f.invalidated_count}, aborted {f.aborted_count})")
        for rep in f.repetitions:
            print(f"  rep {rep.index}: peak {rep.peak_deg:.1f} deg, to-peak "
                  f"{rep.time_to_peak_s:.2f}s, duration {rep.duration_s:.2f}s")  # fmt: skip
        weak = collections.Counter(w for r in mine for w in r["weak_landmarks"].split("|") if w)
        if weak:
            print("not visible enough: " + ", ".join(f"{k} ({v})" for k, v in weak.most_common(5)))
        ang = [float(r["motion_angle"]) for r in mine if r["motion_angle"]]
        if ang:
            confs = [float(r["motion_conf"]) for r in mine if r["motion_angle"]]
            print(f"{mine[0]['movement']} angle: min {min(ang):.1f}, max {max(ang):.1f} deg; "
                  f"median confidence {statistics.median(confs):.2f}")  # fmt: skip
            st = collections.Counter(r["motion_status"] for r in mine)
            print("motion status counts: " + ", ".join(f"{k}={v}" for k, v in st.items()))
            views = collections.Counter(r["view"] for r in mine if r["view"])
            print("camera view seen: " + ", ".join(f"{k}={v}" for k, v in views.most_common()))
        warn = {w for r in mine for w in r["warnings"].split("|") if w}
        print(f"compensation warnings: {sorted(warn) or 'none'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--arm", choices=["right", "left", "both"], default="both")
    ap.add_argument("--torso", choices=["hips", "shoulders"], default="hips")
    ap.add_argument(
        "--movement",
        choices=["none"] + [m.value.lower() for m in ShoulderMovement],
        default="none",
        help="exercise being performed (explicit selection); none = legacy elevation tracking",
    )
    ap.add_argument(
        "--guided",
        action="store_true",
        help="step-by-step session through ALL nine movements (computes all, logs per-step)",
    )
    ap.add_argument("--all-movements", action="store_true", help="compute all nine each frame")
    ap.add_argument("--duration", type=float, default=90.0, help="seconds after countdown")
    ap.add_argument("--countdown", type=float, default=8.0, help="get-in-position seconds")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}")
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mode = TorsoMode.HIPS if args.torso == "hips" else TorsoMode.SHOULDERS_ONLY_ASSUMED_VERTICAL
    arms = ["right", "left"] if args.arm == "both" else [args.arm]
    movement = None if args.movement == "none" else ShoulderMovement(args.movement.upper())
    every = args.guided or args.all_movements
    pipes = {
        a: ShoulderAnalysisPipeline(
            laterality=a, torso_mode=mode, movement=movement, compute_all_movements=every
        )
        for a in arms
    }
    if args.guided:
        args.duration = GUIDED_TOTAL_S
        print(f"GUIDED session: {len(GUIDED)} steps, {GUIDED_TOTAL_S:.0f}s after the countdown.")

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"session_{datetime.now():%Y%m%d_%H%M%S}.csv"
    rows: list[dict[str, str]] = []
    print(f"Camera open. Arms: {arms}. Torso reference: {mode.value}. Movement: {args.movement}.")
    print(f"Countdown {args.countdown:.0f}s, then keep arms hanging still ~3s to calibrate,")
    print("then raise an arm slowly out to the side / forward and back, several times.")

    t0 = time.monotonic()
    last_ts = -1
    frame_i = 0
    prompt: list[str] | None = None
    last_phase: tuple[int, bool] | None = None
    last: dict[str, FrameResult | None] = dict.fromkeys(arms)
    with AuriPoseLandmarker(running_mode=VisionTaskRunningMode.VIDEO) as lm:
        while True:
            ok, bgr = cap.read()
            if not ok:
                print("ERROR: camera read failed")
                break
            elapsed = time.monotonic() - t0
            ts = max(int(elapsed * 1000.0), last_ts + 1)  # strictly increasing for MediaPipe
            last_ts = ts
            started = time.perf_counter()
            # Unmirrored frame -> anatomical LEFT/RIGHT labels stay correct.
            pose = lm.detect_video_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), ts, frame_i)
            counting_down = elapsed < args.countdown
            if counting_down:
                msg = f"Get in position... {args.countdown - elapsed:.0f}"
            else:
                t_in = elapsed - args.countdown
                step = _step_at(t_in) if args.guided else None
                title = prompt_mv = None
                if step:
                    i, left, prep = step
                    name, prompt_mv, _, text = GUIDED[i]
                    head = f"STEP {i + 1}/{len(GUIDED)}: {name}"
                    if prep:  # lead-in: shown, never scored
                        title, prompt_mv = f"{name} [prep]", None
                        prompt = [f"GET READY {left:.0f}s - {head}", *_wrap(text, 40)]
                    else:
                        title = name
                        prompt = [f"DO IT NOW  {left:.0f}s - {head}", *_wrap(text, 40)]
                    if (i, prep) != last_phase:
                        _beep(660 if prep else 988)
                        last_phase = (i, prep)
                title = title or ""
                for arm, pipe in pipes.items():
                    res = pipe.process(pose)
                    last[arm] = res
                    rows.append(_row(arm, float(ts), pose.is_detected, res,
                                     (time.perf_counter() - started) * 1000.0,
                                     title, prompt_mv))  # fmt: skip
                    if res.channel.status == ChannelStatus.CALIBRATION_REJECTED:
                        pipe.recalibrate()  # subject was moving: retry automatically
                msg = f"AURI  t={elapsed - args.countdown:.0f}s  torso: {mode.value.lower()}"
                if (args.guided and step is None) or t_in > args.duration:
                    break
            frame_i += 1

            if not args.no_preview:
                view = cv2.flip(bgr, 1)
                _draw(
                    view,
                    pose,
                    {} if counting_down else last,
                    msg,
                    None if counting_down else prompt,
                )
                cv2.imshow("AURI live (q quit, r recalibrate)", view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    for pipe in pipes.values():
                        pipe.recalibrate()
    cap.release()
    cv2.destroyAllWindows()

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"Numeric log written: {csv_path} (no video/images saved)")
        if args.guided:
            _coverage(rows)
        _summary(rows, pipes, time.monotonic() - t0 - args.countdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
