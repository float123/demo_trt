# demo_trt
This is an example of implementing ctpn using tensorrt.
The example does not use dynamic mode and reshapes the image to (600,1200).
In order to observe tensor more intuitively, the code is not concise.<br>

## Defect
Running demo_org.py and demo_trt.py, you will find that the results obtained by the conv step are basically the same, 
but the results from bilstm are very different.<br>
![](https://github.com/float123/demo_trt/tree/master/CTPN/result.png)

## Environment
python3<br>
tensorflow <= 1.13<br>
tensorrt == 7.0<br>

## Setup
cd ./CTPN/lib/utils/<br>
chmod +x make.sh<br>
./make.sh<br>
python get_data.py<br>

## Testing
The original execution code, the ctpn model in / CTPN / lib / networks / VGGnet_test.py and network.py<br>
python demo_org.py<br>
trt code, model in ctpn_static.py，The execution process is in ctpnport_trt.py<br>
python demo_trt.py<br>
