StockfishNN play_gui.py fix

This fixes the Tkinter TclError: bad screen distance "0 10".
Tkinter Label pady must be a single integer; the tuple (0, 10) is valid for pack/grid geometry options, not the Label constructor option.

Install: copy play_gui.py into your existing StockfishNN folder and replace the old play_gui.py. Do not replace config.json, models, data, or games.
Then run play.bat again.
