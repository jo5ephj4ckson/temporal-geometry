Device Bill of Materials (BOM)

This document provides the complete Bill of Materials required to reproduce the basic entropy‑probe device. All components are standard, inexpensive, and widely available.

1. Active Components

LM358P Operational Amplifier

Type: Dual op‑amp, single‑supply

Package: DIP‑8

Notes: Only Channel A is used; Channel B remains unconnected.

LED Probe

Type: Standard 5 mm LED

Color: Any (forward voltage varies by color)

Forward Voltage: 1.8–3.2 V depending on LED color

Notes: LED anode is used as the probe output node.

2. Passive Components

Bias Resistor

Value: 330 Ω

Power Rating: 1/4 W

Tolerance: ±5%

Purpose: Sets LED current between 8–12 mA.

Decoupling Capacitor

Value: 0.1 µF

Type: Ceramic

Placement: Across VCC and GND near LM358P pins.

3. Power and Connectivity

USB Power Source

Type: Standard USB‑A 5 V supply

Notes: Provides stable 5 V for LED bias and op‑amp power.

3.5 mm TRS Jack

Type: Panel‑mount

Connections:

Tip → LM358P output (Pin 1)

Sleeve → Ground

Ring → Unused

USB Audio Adapter

Model: Sabrent AU‑MMSA or equivalent

Purpose: Captures the analog output signal for digital sampling.

4. Wiring Materials

Hookup Wire

Gauge: 22–26 AWG

Type: Solid or stranded

Notes: Short LED probe leads (<5 cm) recommended.

5. Optional Materials

Breadboard or Perfboard

Purpose: Physical mounting and layout.

Enclosure (Not Required)

Purpose: Protection; not necessary for basic operation.

Summary

This BOM contains all components required to reproduce the basic entropy‑probe device. No additional shielding, filtering, or specialized hardware is required for functional operation.
