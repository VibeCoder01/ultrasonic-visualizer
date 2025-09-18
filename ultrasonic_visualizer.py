#!/usr/bin/env python3
"""Live ultrasonic frequency visualizer.

This script listens to an input microphone, computes a real-time FFT, and
plots only the ultrasonic portion (>20 kHz) of the spectrum. It requires
`sounddevice`, `numpy`, and `matplotlib`.
"""

from __future__ import annotations

import argparse
import sys
import threading
from collections import deque

import numpy as np
import sounddevice as sd

# Matplotlib imports are intentionally deferred to keep startup fast.
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider, Button, TextBox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listen to a microphone and plot ultrasonic frequencies in real time",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Microphone device name or index; omit to use the default input",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=96_000,
        help="Sampling rate in Hz; use a value supported by your hardware",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=4096,
        help="Audio frames per block fetched from the input stream",
    )
    parser.add_argument(
        "--fft-size",
        dest="fft_size",
        type=int,
        default=16_384,
        help="Number of samples used per FFT window",
    )
    parser.add_argument(
        "--min-frequency",
        dest="min_frequency",
        type=float,
        default=20_000.0,
        help="Lower bound of the displayed spectrum, in Hz",
    )
    parser.add_argument(
        "--history",
        type=float,
        default=0.5,
        help="Seconds of audio to retain in the rolling buffer",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    return parser.parse_args()


def list_devices() -> None:
    print(sd.query_devices())


class AudioRingBuffer:
    """Thread-safe ring buffer that stores the latest samples."""

    def __init__(self, max_samples: int) -> None:
        self._buffer = deque(maxlen=max_samples)
        self._lock = threading.Lock()

    def extend(self, data: np.ndarray) -> None:
        with self._lock:
            self._buffer.extend(data)

    def latest(self, count: int) -> np.ndarray:
        with self._lock:
            if not self._buffer:
                return np.zeros(count, dtype=np.float32)
            if len(self._buffer) < count:
                padding = np.zeros(count - len(self._buffer), dtype=np.float32)
                return np.concatenate([padding, np.array(self._buffer, dtype=np.float32)])
            data = list(self._buffer)[-count:]
            return np.array(data, dtype=np.float32)


def build_plot(freqs: np.ndarray, magnitudes: np.ndarray) -> tuple[plt.Figure, plt.Axes, plt.Line2D, dict]:
    """Create plot and UI placeholders.

    Returns (fig, spectrum_ax, line, ui_axes) where ui_axes is a dict of named
    axes slots to host interactive widgets.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.26, top=0.90)

    (line,) = ax.plot(freqs / 1_000, magnitudes, lw=1, color="tab:blue")
    ax.set_title("Ultrasonic Spectrum")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Magnitude (dBFS)")
    ax.set_ylim(-120, 0)
    ax.set_xlim(freqs[0] / 1_000, freqs[-1] / 1_000)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    # Reserve axes for widgets along the bottom area.
    ui_axes = {
        "minfreq": plt.axes([0.10, 0.18, 0.56, 0.035]),
        "fft": plt.axes([0.10, 0.13, 0.56, 0.035]),
        "block": plt.axes([0.10, 0.08, 0.56, 0.035]),
        "device_box": plt.axes([0.70, 0.18, 0.18, 0.05]),
        "samplerate_box": plt.axes([0.70, 0.12, 0.18, 0.05]),
        "apply_device": plt.axes([0.90, 0.18, 0.07, 0.05]),
        "list_devices": plt.axes([0.90, 0.12, 0.07, 0.05]),
        "pause": plt.axes([0.90, 0.06, 0.07, 0.05]),
    }

    return fig, ax, line, ui_axes


def main() -> None:
    args = parse_args()

    if args.list_devices:
        list_devices()
        return

    try:
        sd.check_input_settings(device=args.device, samplerate=args.samplerate)
    except Exception as exc:  # pragma: no cover - interactive diagnostic
        print(f"Failed to open audio device: {exc}", file=sys.stderr)
        sys.exit(1)

    # Shared state used by callbacks
    state: dict[str, object] = {}

    ring = AudioRingBuffer(max_samples=int(args.history * args.samplerate))

    def compute_fft_params(samplerate: int, fft_size: int, min_frequency: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        window = np.hanning(fft_size)
        full_freqs = np.fft.rfftfreq(fft_size, d=1.0 / samplerate)
        mask = full_freqs >= min_frequency
        display_freqs = full_freqs[mask]
        return window, full_freqs, mask, display_freqs

    fft_window, full_freqs, ultrasonic_mask, display_freqs = compute_fft_params(
        args.samplerate, args.fft_size, args.min_frequency
    )

    def audio_callback(indata: np.ndarray, frames: int, time, status) -> None:  # pragma: no cover
        if status:
            print(status, file=sys.stderr)
        mono = np.mean(indata, axis=1)
        ring.extend(mono)

    def start_stream(device: str | int | None, samplerate: int, blocksize: int) -> sd.InputStream:
        stream = sd.InputStream(
            device=device,
            channels=1,
            samplerate=samplerate,
            blocksize=blocksize,
            callback=audio_callback,
        )
        stream.start()
        return stream

    def stop_stream(stream: sd.InputStream | None) -> None:
        if stream is None:
            return
        try:
            stream.stop()
        finally:
            stream.close()

    # Start stream
    stream = start_stream(args.device, args.samplerate, args.blocksize)
    state["stream"] = stream

    # Initial spectrum for plotting
    initial_samples = ring.latest(args.fft_size)
    windowed = initial_samples * fft_window
    spectrum = np.fft.rfft(windowed)
    magnitudes = 20.0 * np.log10(np.maximum(np.abs(spectrum), 1e-12))

    fig, ax, line, ui_axes = build_plot(display_freqs, magnitudes[ultrasonic_mask])

    # Info banner text
    info_text = fig.text(0.01, 0.94, "", ha="left", va="center", fontsize=9)

    def device_name(dev: str | int | None) -> str:
        try:
            if dev is None:
                return sd.query_devices(kind="input")["name"]
            if isinstance(dev, int) or (isinstance(dev, str) and dev.isdigit()):
                dev_index = int(dev)
                return f"{dev_index}: {sd.query_devices(dev_index)['name']}"
            # string name or pattern
            return str(dev)
        except Exception:
            return str(dev)

    def update_info_banner() -> None:
        try:
            s: sd.InputStream = state.get("stream")  # type: ignore[assignment]
            dev = args.device
            # When stream is active, prefer its resolved device and samplerate
            current_sr = int(s.samplerate) if s else args.samplerate
            current_block = int(s.blocksize) if s and s.blocksize is not None else args.blocksize
            latency = getattr(s, "latency", None)
            nyquist = current_sr / 2.0
            bin_hz = current_sr / args.fft_size
            msg = (
                f"Device: {device_name(dev)}  |  SR: {current_sr/1000:.1f} kHz  "
                f"|  Block: {current_block}  |  FFT: {args.fft_size} (Δf≈{bin_hz:.1f} Hz)  "
                f"|  Min: {args.min_frequency/1000:.1f} kHz  "
                f"|  Nyquist: {nyquist/1000:.1f} kHz"
            )
            if latency is not None:
                try:
                    lat_ms = (
                        latency[0] * 1000.0 if isinstance(latency, (list, tuple)) else float(latency) * 1000.0
                    )
                    msg += f"  |  Latency≈{lat_ms:.1f} ms"
                except Exception:
                    pass
            info_text.set_text(msg)
        except Exception as exc:  # pragma: no cover - best effort UI
            info_text.set_text(f"Info unavailable: {exc}")

    update_info_banner()

    # Widgets
    minfreq_slider = Slider(
        ui_axes["minfreq"],
        label="Min Freq (kHz)",
        valmin=10.0,
        valmax=float(args.samplerate) / 2000.0,
        valinit=args.min_frequency / 1000.0,
        valstep=0.5,
    )

    fft_log2_init = int(np.log2(args.fft_size))
    fft_slider = Slider(
        ui_axes["fft"],
        label="FFT Size (2^n)",
        valmin=11,
        valmax=18,
        valinit=fft_log2_init,
        valstep=1,
    )

    block_slider = Slider(
        ui_axes["block"],
        label="Blocksize",
        valmin=256,
        valmax=8192,
        valinit=args.blocksize,
        valstep=256,
    )

    device_box = TextBox(ui_axes["device_box"], "Device", initial=str(args.device or "default"))
    sr_box = TextBox(ui_axes["samplerate_box"], "SR (Hz)", initial=str(args.samplerate))
    apply_btn = Button(ui_axes["apply_device"], "Apply\nDevice")
    list_btn = Button(ui_axes["list_devices"], "List\nDevs")
    pause_btn = Button(ui_axes["pause"], "Pause")

    # State variables for interaction
    is_paused = {"value": False}

    def set_fft_size(n_fft: int) -> None:
        args.fft_size = int(n_fft)
        nonlocal fft_window, full_freqs, ultrasonic_mask, display_freqs
        try:
            current_sr = int(float(sr_box.text))
        except ValueError:
            current_sr = args.samplerate
        fft_window, full_freqs, ultrasonic_mask, display_freqs = compute_fft_params(
            current_sr, args.fft_size, args.min_frequency
        )
        # Update x-axis data and limits
        line.set_xdata(display_freqs / 1000.0)
        ax.set_xlim(display_freqs[0] / 1000.0, display_freqs[-1] / 1000.0)
        update_info_banner()

    def set_min_frequency(min_f: float) -> None:
        args.min_frequency = float(min_f)
        nonlocal ultrasonic_mask, display_freqs
        ultrasonic_mask = full_freqs >= args.min_frequency
        display_freqs = full_freqs[ultrasonic_mask]
        line.set_xdata(display_freqs / 1000.0)
        ax.set_xlim(display_freqs[0] / 1000.0, display_freqs[-1] / 1000.0)
        update_info_banner()

    def reopen_stream(new_device: str | int | None, new_sr: int, new_block: int) -> None:
        nonlocal stream, ring
        try:
            sd.check_input_settings(device=new_device, samplerate=new_sr)
        except Exception as exc:  # pragma: no cover - interactive validation
            print(f"Audio settings invalid: {exc}", file=sys.stderr)
            return
        stop_stream(stream)
        stream = start_stream(new_device, new_sr, new_block)
        state["stream"] = stream
        args.device = new_device  # keep args in sync with UI
        args.samplerate = new_sr
        args.blocksize = new_block
        # Resize ring to match the new samplerate history length
        ring = AudioRingBuffer(max_samples=int(args.history * new_sr))
        # Recompute FFT dependent params
        nonlocal fft_window, full_freqs, ultrasonic_mask, display_freqs
        fft_window, full_freqs, ultrasonic_mask, display_freqs = compute_fft_params(
            new_sr, args.fft_size, args.min_frequency
        )
        # Update slider bounds that depend on samplerate (Nyquist)
        minfreq_slider.valmax = new_sr / 2000.0
        minfreq_slider.ax.set_xlim(minfreq_slider.valmin, minfreq_slider.valmax)
        if minfreq_slider.val > minfreq_slider.valmax:
            minfreq_slider.set_val(minfreq_slider.valmax)
        # Update x-axis domain
        line.set_xdata(display_freqs / 1000.0)
        ax.set_xlim(display_freqs[0] / 1000.0, display_freqs[-1] / 1000.0)
        update_info_banner()

    # Wire up widget callbacks
    def _on_minfreq(val: float) -> None:
        set_min_frequency(val * 1000.0)

    def _on_fft(val: float) -> None:
        n_fft = int(2 ** int(round(val)))
        set_fft_size(n_fft)

    def _on_block(val: float) -> None:
        new_block = int(val)
        reopen_stream(device_box.text.strip() or None, int(float(sr_box.text)), new_block)

    def _on_apply(_event) -> None:
        # Allow device by index or name substring. If a non-digit string is
        # provided, we pass it through and PortAudio will resolve a match.
        dev_text = device_box.text.strip()
        dev: str | int | None
        if dev_text.lower() in {"default", ""}:
            dev = None
        elif dev_text.isdigit():
            dev = int(dev_text)
        else:
            dev = dev_text
        try:
            sr = int(float(sr_box.text))
        except ValueError:
            print(f"Invalid samplerate: {sr_box.text}", file=sys.stderr)
            sr = args.samplerate
            sr_box.set_val(str(sr))
        reopen_stream(dev, sr, int(block_slider.val))

    def _on_list(_event) -> None:
        # Print a concise device table to stdout
        try:
            devices = sd.query_devices()
            print("\nInput devices:")
            for i, d in enumerate(devices):
                if int(d.get("max_input_channels", 0)) > 0:
                    print(f"  {i:2d}: {d['name']}  (in/out: {d['max_input_channels']}/{d['max_output_channels']})")
        except Exception as exc:  # pragma: no cover
            print(f"Failed to list devices: {exc}")

    def _on_pause(_event) -> None:
        is_paused["value"] = not is_paused["value"]
        pause_btn.label.set_text("Resume" if is_paused["value"] else "Pause")

    minfreq_slider.on_changed(_on_minfreq)
    fft_slider.on_changed(_on_fft)
    block_slider.on_changed(_on_block)
    apply_btn.on_clicked(_on_apply)
    list_btn.on_clicked(_on_list)
    pause_btn.on_clicked(_on_pause)

    def update_plot(_frame: int):  # pragma: no cover - animation loop
        if is_paused["value"]:
            return (line,)
        samples = ring.latest(args.fft_size)
        windowed_samples = samples * fft_window
        spectrum = np.fft.rfft(windowed_samples)
        magnitudes = 20.0 * np.log10(np.maximum(np.abs(spectrum), 1e-12))
        line.set_ydata(magnitudes[ultrasonic_mask])
        return (line,)

    ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=True)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        stop_stream(stream)


if __name__ == "__main__":
    main()

