# Wiring_Map

This document specifies the exact pin‑level wiring for the entropy‑probe device using the LM358P op‑amp and associated components.

---

## 1. LM358P Pin Assignments

- **Pin 1 – Output A:**  
  Connected to **TRS jack tip** (signal output).

- **Pin 2 – Inverting Input A (−):**  
  Connected to **Pin 1 (Output A)** for simple voltage‑follower / buffer configuration.

- **Pin 3 – Non‑inverting Input A (+):**  
  Connected to the **LED probe node** (LED anode).

- **Pin 4 – GND:**  
  Connected to **USB ground** and **TRS jack sleeve**.

- **Pin 5 – Non‑inverting Input B (+):**  
  Unused; leave unconnected.

- **Pin 6 – Inverting Input B (−):**  
  Unused; leave unconnected.

- **Pin 7 – Output B:**  
  Unused; leave unconnected.

- **Pin 8 – VCC:**  
  Connected to **USB +5 V**.

---

## 2. LED Probe Wiring

- **LED Anode:**  
  Connected to **330 Ω resistor**, then to **USB +5 V**.  
  Also connected directly to **LM358P Pin 3 (Non‑inverting input A)** — this junction is the **probe output node**.

- **LED Cathode:**  
  Connected to **USB ground**.

---

## 3. Power Wiring

- **USB +5 V:**  
  - To **LM358P Pin 8 (VCC)**  
  - To **330 Ω resistor** (LED bias)

- **USB Ground:**  
  - To **LM358P Pin 4 (GND)**  
  - To **LED cathode**  
  - To **TRS jack sleeve**

- **Decoupling Capacitor (0.1 µF):**  
  - One lead to **Pin 8 (VCC)**  
  - One lead to **Pin 4 (GND)**  
  - Placed physically close to the LM358P.

---

## 4. TRS Jack Wiring (3.5 mm)

- **Tip:**  
  Connected to **LM358P Pin 1 (Output A)**.

- **Ring:**  
  Unused; leave unconnected.

- **Sleeve:**  
  Connected to **USB ground** and **LM358P Pin 4 (GND)**.

---

## 5. USB Audio Adapter Connection

- **TRS plug tip:** → Sabrent AU‑MMSA line‑in tip  
- **TRS plug sleeve:** → Sabrent AU‑MMSA ground

No additional components are required between the op‑amp output and the USB audio adapter.

---

## Summary

This wiring map defines all electrical connections needed to reproduce the entropy‑probe device exactly. No hidden nodes or implicit connections are assumed; all signal, power, and ground paths are explicitly specified.
