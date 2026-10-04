# Probe_Spec

This document defines the electrical and functional characteristics of the LED‑based tunneling probe used in the entropy‑capture device. All parameters are explicit to ensure reproducibility.

---

## 1. Probe Overview

The probe is a single LED biased from a 5 V USB supply through a 330 Ω resistor.  
The **probe output node** is the LED **anode**, which exhibits micro‑scale temporal fluctuations suitable for entropy sampling when buffered through the LM358P.

---

## 2. Electrical Characteristics

### LED Type
- **Package:** 5 mm through‑hole LED  
- **Color:** Any (red, green, blue, etc.)  
- **Forward Voltage (Vf):**  
  - Red: ~1.8–2.0 V  
  - Green: ~2.0–2.2 V  
  - Blue/White: ~3.0–3.2 V  
- **Forward Current:** 8–12 mA (set by resistor)

### Bias Resistor
- **Value:** 330 Ω  
- **Power Rating:** 1/4 W  
- **Tolerance:** ±5%  
- **Function:** Sets LED current and stabilizes probe node voltage.

### Operating Voltage
- **Supply:** 5 V USB  
- **Probe Node Voltage:** Vf + (I × R) dynamics; typically 1.8–3.2 V depending on LED type.

---

## 3. Node Definitions

### Probe Output Node
- **Location:** LED anode  
- **Connections:**  
  - To LM358P Pin 3 (non‑inverting input A)  
  - To 330 Ω resistor → USB +5 V  
- **Behavior:**  
  - Exhibits micro‑scale temporal fluctuations  
  - Buffered by LM358P to produce stable analog output  
  - No filtering or additional conditioning required

### Ground Reference
- **Location:** LED cathode  
- **Connections:**  
  - USB ground  
  - LM358P Pin 4 (GND)

---

## 4. Physical Layout Requirements

- LED leads should be **short (<5 cm)** to minimize parasitic capacitance.  
- LED may be mounted vertically or horizontally; orientation does not affect function.  
- No shielding, enclosure, or thermal stabilization is required for basic operation.

---

## 5. Expected Output Characteristics

- **Amplitude:** 0.1–1.2 V after LM358P buffering  
- **Frequency Content:** Broadband micro‑fluctuations suitable for entropy extraction  
- **Sampling:** 44.1 kHz or 48 kHz via Sabrent AU‑MMSA

---

## Summary

This specification defines the LED probe’s electrical behavior, wiring, and operating conditions. When combined with the Wiring_Map and Device_BOM, it provides complete reproducibility for the entropy‑probe hardware.
