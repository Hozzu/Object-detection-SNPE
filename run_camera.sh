export LD_LIBRARY_PATH=lib/:snpe-1.58/lib/
export ADSP_LIBRARY_PATH=snpe-1.58/lib/

./pkshin_detect camera model/ssd_mobilenet_v2.dlc coco_labels.txt ssdDisplay.xml npu
