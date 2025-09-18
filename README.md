Ultrasonic Visualizer
=====================

Real-time ultrasonic spectrum display using your microphone. Captures audio via PortAudio (sounddevice), computes an FFT, and plots only the ultrasonic band (>20 kHz). Includes a control panel to adjust device, sample rate, block size, min frequency, and FFT size while running.

Features
--------
- Live FFT with Hann window and dBFS scaling
- Adjustable min frequency, FFT size (2^11–2^18), and block size
- Device selection by index or name; change sample rate on the fly
- Pause/resume; info banner with Δf, Nyquist, latency

Requirements
------------
- Python 3.11+
- `numpy`, `matplotlib`, `sounddevice`
- PortAudio runtime (Linux): `sudo apt-get install -y libportaudio2`
- GUI backend (recommended): `sudo apt-get install -y python3-tk`

Quick Start
-----------
1) Create a venv and install deps:

   python3 -m venv .venv && source .venv/bin/activate
   pip install -U pip
   pip install -e .

2) Run it:

   # list audio input devices
   python ultrasonic_visualizer.py --list-devices

   # launch with specific device and samplerate
   python ultrasonic_visualizer.py --device "hw:0,0" --samplerate 96000

   # or after installation: ultrasonic-visualizer --device "hw:0,0" --samplerate 96000

GUI Controls
------------
- Min Freq (kHz): lower bound of displayed spectrum
- FFT Size (2^n): frequency resolution (Δf = SR / N)
- Blocksize: audio callback buffer size (latency vs. update rate)
- Device, SR (Hz): change input and sample rate; click Apply Device
- List Devs: print available device indices/names to the terminal
- Pause: freeze/unfreeze the plot

Troubleshooting
---------------
- PortAudio not found: `sudo apt-get install -y libportaudio2`
- No GUI window: install `python3-tk` or use X/Wayland forwarding
- Roll-off around 20–24 kHz: likely your input path is 48 kHz (Nyquist ~24 kHz) or hardware/mic low-pass filtering

License
-------
TBD by repository owner.
