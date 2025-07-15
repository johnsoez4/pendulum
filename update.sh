#! /bin/bash

pixi self-update
pixi clean cache --conda
pixi update
pixi clean cache --conda
pip cache purge
(cd ~/.cache/rattler/ && rm -rf *)
pixi shell
