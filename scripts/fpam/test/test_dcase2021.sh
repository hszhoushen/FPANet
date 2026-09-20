#!/bin/bash

# 运行第一个.sh文件
sh ./scripts/fpam/test/test_fpam_dcase2021_audio.sh
wait

# 运行第二个.sh文件
sh ./scripts/fpam/test/test_fpam_dcase2021_visual.sh
wait

# 运行第三个.sh文件
sh ./scripts/fpam/test/test_fpam_dcase2021_audio_visual.sh
wait