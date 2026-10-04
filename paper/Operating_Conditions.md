# Operating_Conditions

This document defines the electrical, environmental, and sampling conditions required for correct operation of the entropy‑probe device. All parameters are explicit to ensure reproducibility.

---

## 1. Power Conditions

### Supply Voltage
- **Source:** Standard USB‑A port  
- **Voltage:** 5.0 V ± 5%  
- **Current Draw:** < 20 mA total (LED + LM358P)

### Stability Requirements
- No regulated bench supply required.  
- No additional filtering beyond the 0.1 µF decoupling capacitor.

---

## 2. LED Probe Operating Range

### LED Current
- **Typical:** 8–12 mA  
- **Set By:** 330 Ω resistor at 5 V supply

### Probe Node Voltage
- **Range:** 1.8–3.2 V depending on LED color  
- **Behavior:** Micro‑scale temporal fluctuations suitable for entropy extraction

---

## 3. Op‑Amp Operating Conditions (LM358P)

### Supply
- **VCC:** 5 V  
- **GND:** USB ground

### Output Characteristics
- **Amplitude:** 0.1–1.2 V typical  
- **Output Impedance:** Low; suitable for direct line‑in connection  
- **Configuration:** Voltage follower (buffer)

### Stability
- No oscillation observed under standard wiring.  
- Decoupling capacitor must be placed close to Pins 4 and 8.

---

## 4. Audio Capture Conditions

### USB Audio Adapter
- **Model:** Sabrent AU‑MMSA or equivalent  
- **Input:** Line‑in (not microphone‑biased)

### Sampling Rates
- **Supported:** 44.1 kHz or 48 kHz  
- **Bit Depth:** 16‑bit recommended

### Expected Signal
- Broadband analog fluctuations  
- No DC‑blocking or filtering required  
- No gain adjustment required; default line‑in gain is sufficient

---

## 5. Environmental Conditions

### Temperature
- **Operating Range:** 0–40 °C  
- No thermal stabilization required.

### Electromagnetic Environment
- Device operates correctly in typical indoor environments.  
- No shielding or enclosure required for basic operation.

### Mechanical
- LED leads should be kept short (< 5 cm).  
- Breadboard or perfboard mounting is sufficient.

---

## 6. Startup and Runtime Behavior

### Startup
- Device becomes operational immediately upon USB power application.  
- No warm‑up period required.

### Runtime
- Continuous operation stable for multi‑hour sampling sessions.  
- No drift correction or recalibration required.

---

## Summary

These operating conditions define the complete electrical, environmental, and sampling requirements for the entropy‑probe device. When combined with the BOM, Wiring Map, and Probe Specification, the device is fully reproducible and requires no additional assumptions.
