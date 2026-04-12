---
source: owner-manual.pdf
page: 45
image: owner-manual_p045.png
content_type: schematic
section: parts
---

# Page 45: Wiring Schematic

Full electrical wiring schematic of the Vulcan OmniPro 220 welder. This is a complex circuit diagram showing the complete internal electrical architecture.

## Major Components Identified in Schematic:

### Power Input (Bottom-Left):
- **K1 AC 120-240V 50/60Hz** -- main power input relay/contactor
- **AC1** and **AC2** -- AC input lines
- Ground symbol connected to chassis

### Power Factor Correction:
- **PFC** -- Power Factor Correction circuit, shown as a large block in the center-left area
- Connected to the input transformer and rectifier stages

### Power Conversion:
- **IGBT** -- Insulated-Gate Bipolar Transistor block, the main power switching stage. Located in the center of the schematic. This is the inverter that converts DC back to high-frequency AC for the welding transformer.
- Multiple diode bridge components visible in the rectifier stages

### Control Electronics:
- **MCU BOARD** -- Main Control Unit board, located in the center-right area. This is the brain of the welder, containing the microcontroller that manages all welding parameters, LCD display, and user inputs.
- **REMOTE BOARD** -- Secondary control board below the MCU board. Handles remote control signals and wire feed control.
- **LCD SCREEN** -- Connected to the MCU board, shown in the upper-right area.

### Output Section (Top):
- **OUT+** -- Positive output terminal (top-left)
- **OUT-** -- Negative output terminal (top-right)
- Output inductor/choke between the inverter output and the terminals

### Wire Feed System:
- **M** -- Wire feed motor, shown connected to the Remote Board
- **WIRE FEEDER** -- wire feed assembly
- **FAST WIRE FEED SWITCH** -- the cold wire feed switch

### Gas Control:
- **SOLENOID VALVE** -- two solenoid valves shown, connected to the Remote Board. These control shielding gas flow.

### Cooling:
- **FAN** and **FAN2** -- two cooling fans shown at the bottom-center, connected to the power supply

### Connector Labels:
Multiple CN (connector) labels throughout: CN1 through CN14+, representing the internal wiring harness connections between boards and components.

## Note:
This schematic is intended for qualified technicians performing repairs. It shows the complete signal and power path from AC input through power conversion to welding output, plus all control, display, and auxiliary circuits.
