# Menu button artwork

Drop a PNG here named after its label and the menu uses it instead of the
generated plate:

| button | filename |
|---|---|
| START | `start.png` |
| TRAINING | `training.png` |
| SETTINGS | `settings.png` |
| QUIT | `quit.png` |
| APPLY CAMERA | `apply-camera.png` |
| BACK TO MENU | `back-to-menu.png` |

The rule is `spidergame.render3d.buttonart.slug()`: lowercase, non-alphanumerics
become hyphens, runs collapse.

**These files carry their own baked-in label**, so a button backed by one draws
no text of its own — otherwise the label would render twice. That also means
its centring is whatever the exporter baked in, and selection can only be shown
as a brightness change over the fixed image.

With no file present the button is drawn procedurally in the same style (red
plate, heavy blue outline, rounded corners) with the label rendered live and
centred against the plate's true middle. That path is the one with exact
alignment at any size, so prefer it unless the artwork does something the
generator cannot.
