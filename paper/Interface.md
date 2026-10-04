# Interface

This document defines the exact electrical and physical interface between the entropy‑probe hardware and the USB audio capture device. All signal paths, connector mappings, and expected levels are explicitly specified.

---

## 1. Output Connector (3.5 mm TRS)

The device uses a standard **3.5 mm TRS jack** for analog output.

### TRS Pin Mapping
- **Tip:**  
  LM358P Output A (Pin 1) — primary signal output.

- **Ring:**  
  Unused — leave unconnected.

- **Sleeve:**  
  Ground — shared with USB ground and LM358P Pin 4.

### Notes
- The TRS jack is wired as a **mono output**.  
- No additional filtering, buffering, or impedance matching is required.

---

## 2. USB Audio Adapter (Sabrent AU‑MMSA)

The TRS output connects directly to the **Sabrent AU‑MMSA** USB audio adapter (or equivalent).

### Connection
- **TRS plug tip → AU‑MMSA line‑in tip**  
- **TRS plug sleeve → AU‑MMSA ground**

### Adapter Requirements
- Must support **44.1 kHz or 48 kHz** sampling.  
- Must expose a **line‑in** input (not microphone‑biased).

---

## 3. Expected Signal Characteristics

### Voltage Levels
- **Typical amplitude:** 0.1–1.2 V peak  
- **DC offset:** Determined by LED probe node voltage (1.8–3.2 V), internally buffered by LM358P.

### Impedance
- LM358P output impedance is low enough for direct connection to AU‑MMSA line‑in.  
- No series resistor or coupling capacitor is required.

### Signal Type
- Broadband micro‑fluctuation analog signal suitable for entropy extraction.  
- No modulation, encoding, or filtering applied.

---

## 4. Grounding

All grounds are common:
- USB ground  
- LM358P Pin 4  
- LED cathode  
- TRS sleeve  
- AU‑MMSA ground

This ensures consistent reference levels and avoids ground‑loop artifacts.

---

## 5. Physical Interface Notes

- Use a standard **3.5 mm TRS male‑to‑male cable**.  
- Cable length is not critical; typical 1–2 m cables work without degradation.  
- No shielding or ferrite beads are required for basic operation.

---

## Summary

This interface specification defines the complete analog output pathway from the entropy‑probe device to the USB audio adapter. With this document, the output stage is fully reproducible and requires no additional assumptions or hidden wiring.
