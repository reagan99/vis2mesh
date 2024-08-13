# Vis2Mesh

This is the offical repository of the paper:

**Vis2Mesh: Efficient Mesh Reconstruction from Unstructured Point Clouds of Large Scenes with Learned Virtual View Visibility**

[ICCV](https://openaccess.thecvf.com/content/ICCV2021/html/Song_Vis2Mesh_Efficient_Mesh_Reconstruction_From_Unstructured_Point_Clouds_of_Large_ICCV_2021_paper.html) | [Arxiv](https://arxiv.org/abs/2108.08378) | [Presentation](https://youtu.be/KN55e_q6llE?feature=shared)

# 다음 순서로 진행
#### 깃 클론 위치는 home/{계정명}/ 이라고 가정

Conda 파이썬 3.8로 설치 및 아래 작업은 전부 conda 환경 내에서 진행 - 패키지 설치 포함

파일 명시 이름 변경: 
	find . -type f -exec sed -i 's/sxsong1207/(본인 아이디)/g' {} +


  
## 라이브러리 설치 
Conan ==1.53.0으로 설치 - Conan 프로필 생성 (여러 사용자가 접근 시 오류 발생 하는 것 같음)

Matplotlib 3.5로 설치

Opencv 설치 후 환경변수 설정


	export CPATH=/usr/local/include/opencv4:$CPATH

	export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH

	export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
Cuda 및 여러 설치:


	conda install pytorch == 1.12.1 torchvision == 0.13.1 torchaudio == 0.12.1 cudatoolkit=10.2 -c pytorch


Cudnn 설치 : 

	conda install -c anaconda cudnn


Cuda 툴킷 재설치 - 개발자용: 

	conda install cudatoolkit=10.2 -c hcc

  

재부팅


  
# 빌드 진행
## vvmesh
Tools/build_vvmesh.sh 실행

  



  
## openmvs
Tools/build_openvms.sh 실행

  

Zlib 설치 링크 변경:

  

	find . -type f -exec sed -i 's|http://zlib.net/zlib-1.2.11.tar.gz|http://zlib.net/zlib-1.3.1.tar.gz|g' {} +

  

Tools/build_vvmwsh.sh 재실행 
(한번 openmvs.sh를 실행해 줘야지 링크를 변경 할 수 있고 이후 정상 설치 가능)

 ## 실행 파일 수정

Inference.py, model폴더 내에 있는 파이썬 파일 (cascademodel.py, vvvnet.py) 
tools/lib폴더 내 있는 파이썬 파일 (network_predict.py, hpr_predict.py) 상단에 다음 코드 추가. (깃 클론 위치)


	import sys 
	sys.path.append('/home/{계정명}/vis2mesh/')

  

### 예제파일 및 가중치 다운로드

	./checkpoints/get_pretrained.sh
	./example/get_example.sh

### Inference.py의 findToolset 함수 부분에서 빌드한 도구 직접 경로 명시 

	def  findToolset(dep_tools  = ['vvtool','o3d_vvcreator.py','ReconstructMesh']):
		Toolset=dict()
		print('#Available Toolset#')
		for t in dep_tools:
			Toolset['vvtool'] =  '/home/{계정명}/vis2mesh/tools/bin/vvtool'
			Toolset['o3d_vvcreator.py'] =  '/home/{계정명}/vis2mesh/tools/bin/o3d_vvcreator.py'
			Toolset['ReconstructMesh'] =  '/home/{계정명}/vis2mesh/tools/bin/OpenMVS/ReconstructMesh'
			assert(Toolset[t]!=None)
		print(f'{t}: {Toolset[t]}')
		return Toolset
	
### 가중치 파일 위치 갱신
inference.py 에서 --checkpoint 부분에서 지정하고 있는 가중치 파일 위치 갱신
 	
	visgroup.add_argument('--checkpoint',default='/home/{계정명}/vis2mesh/checkpoints/VDVNet_CascadePPP_epoch30.pth',type=str)
	
## 실행
이후 inferencr.py 실행. 단, --cam 옵션 없이 줄 시 xhost, GLSL 등의 문제로 직접 서버에 모니터를 연결해서 해야함 (원격 데스크 톱 불가능, 다른 방법이 있을 수는 있음)

##### Run with Customize Views

`python inference.py example/example1.ply`
Run the command without `--cam` flag, you can add virtual views interactively with the following GUI. Your views will be recorded in `example/example1.ply_WORK/cam*.json`.

![Main View](imgs/simplecamcreator.png)

Navigate in 3D viewer and click key `[Space]` to record current view. Click key `[Q]` to close the window and continue meshing process.

![Record Virtual Views](imgs/simplecam_recordview.png)


<!-- # Training

export DATASET_PATH="$PWD/dataset"
export PYTHONPATH="$PWD:$PYTHONPATH"

python trainer/train_visnet.py -d $DATASET_PATH -l 0.005 -b 15 -e 50 --decay-step 5 --decay-rate 0.5 -a 'PartialConvUNet(input_channels=2)'
python trainer/train_depthnet.py -d $DATASET_PATH -l 0.005 -b 15 -e 50 --decay-step 5 --decay-rate 0.5 -a 'PartialConvUNet(input_channels=2)'
python trainer/train_depthnetrefine.py -d $DATASET_PATH -l 0.001 -b 8 -e 30 --decay-step 5 --decay-rate 0.5 -a 'PartialConvUNet(input_channels=2)' 'PartialConvUNet(input_channels=2)' -t True True --load0 "checkpoints/VIS_PartialConvUNet(input_channels=2)_epoch50.pth" --load1 "checkpoints/DEPTH_PartialConvUNet(input_channels=2)_epoch50.pth"

mv "checkpoints/REFDEPTH_PartialConvUNet(input_channels=2)_epoch30.pth" "checkpoints/REFDEPTH_PartialConvUNet(input_channels=2)_epoch30_best.pth"

python trainer/train_cascadenet.py -d $DATASET_PATH -l 0.001 -b 5 -e 70 --decay-step 7 --decay-rate 0.5 -a 'PartialConvUNet(input_channels=2)' 'PartialConvUNet(input_channels=2)' 'PartialConvUNet(input_channels=3)' -t False False True --load0 "checkpoints/REFVIS_PartialConvUNet(input_channels=2)_epoch30_best.pth" --load1 "checkpoints/REFDEPTH_PartialConvUNet(input_channels=2)_epoch30_best.pth"

python trainer/train_cascadenet.py -d $DATASET_PATH -l 0.0001 -b 5 -e 30 --decay-step 8 --decay-rate 0.5 -a 'PartialConvUNet(input_channels=2)' 'PartialConvUNet(input_channels=2)' 'PartialConvUNet(input_channels=3)' -t True True True --load "checkpoints/VISDEPVIS_['PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=3)']_epoch70.pth"

mv "checkpoints/VISDEPVIS_['PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=3)']_epoch30.pth" "checkpoints/VISDEPVIS_CascadePPP_epoch30.pth" -->
