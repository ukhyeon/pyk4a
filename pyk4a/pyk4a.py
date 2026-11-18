import numpy as np
from typing import Tuple
import k4a_module
from enum import Enum
import time
from collections import defaultdict
import json
import operator
import threading
import queue

from pyk4a.config import Config, ColorControlMode, ColorControlCommand


# k4a_wait_result_t
class Result(Enum):
    Success = 0
    Failed = 1
    Timeout = 2


class K4AException(Exception):
    pass


class K4ATimeoutException(K4AException):
    pass


class PyK4A:
    TIMEOUT_WAIT_INFINITE = -1

    def __init__(self, config=Config(), device_id=0):
        self._device_id = device_id
        self._config = config
        self.is_running = False
        self.timings = defaultdict(list)
        self.skip_counts = []

    def __del__(self):
        if self.is_running:
            self.disconnect()

    def connect(self, lut=False):
        print('connecting...')
        self._device_open()
        self._start_cameras(lut)
        self.is_running = True
        self.counter = -1
        print('connected')

    def disconnect(self):
        self._stop_cameras()
        self._device_close()
        self.is_running = False

    def _device_open(self):
        res = k4a_module.device_open(self._device_id)
        self._verify_error(res)

    def _device_close(self):
        res = k4a_module.device_close()
        self._verify_error(res)

    def _start_cameras(self, lut=False):
        if lut:
            res = k4a_module.device_start_cameras_with_lut(*self._config.unpack())    
        else:
            res = k4a_module.device_start_cameras(*self._config.unpack())
        self._verify_error(res)

    def _stop_cameras(self):
        res = k4a_module.device_stop_cameras()
        self._verify_error(res)

    def get_capture(self, timeout=TIMEOUT_WAIT_INFINITE, color_only=False, transform_depth_to_color=True):
        """Get next capture
            Parameters:
                timeout: Timeout of capture
                color_only: If True, only color image will be returned
                transform_depth_to_color: If True, depth image will be transformed to the color image space
            Returns:
                color: Color image
                deoth: Depth image
                pose: Pose data of shape (num_bodies, num_joints, 10). Last dimension means:
                      0:2 - 2d keypoints projected to color image
                      2:5 - 3d keypoints in world depth image coordinates
                      5:9 - orientation
                      9 - confidence
                body_index_map: Body Index map is the body instance segmentation map. Each pixel maps to the corresponding pixel in the depth image or the ir image.
                                The value for each pixel represents which body the pixel belongs to. It can be either background (value 255)
                                or the index of a detected k4abt_body_t.
        """

        res = k4a_module.device_get_capture(timeout)
        self._verify_error(res)

        color = k4a_module.device_get_color_image()
        if color_only:
            return color

        # depth = k4a_module.device_get_depth_image(transform_depth_to_color)
        # pose, body_index_map = k4a_module.device_get_pose_data()

        return color, depth, pose, body_index_map

    def get_cam_params(self):
        res = k4a_module.device_get_cam_params()
        res = json.loads(res)
        return res

    def get_serial_number(self):
        get_serial_number = k4a_module.device_get_serial_number()
        return get_serial_number

    def get_capture2(
        self,
        skip_old_atol_ms=None, parallel_bt=True,
        get_bt=True, get_depth=False, get_color=True,
        get_color_timestamp=True, get_depth_timestamp=True,
        undistort_color=False, undistort_depth=True, undistort_bt=False,
        transformed_depth=True,
        timeout=TIMEOUT_WAIT_INFINITE, verbose=0
    ):
        # values
        #     print("[get_capture2] called with arguments:")
        #     print({
        #         "skip_old_atol_ms": skip_old_atol_ms,         # None 
        #         "parallel_bt": parallel_bt,                   # True
        #         "get_bt": get_bt,                             # True
        #         "get_depth": get_depth,                       # False
        #         "get_color": get_color,                       # True
        #         "get_color_timestamp": get_color_timestamp,   # True
        #         "get_depth_timestamp": get_depth_timestamp,   # True
        #         "undistort_color": undistort_color,           # False
        #         "undistort_depth": undistort_depth,           # True
        #         "undistort_bt": undistort_bt,                 # False
        #         "transformed_depth": transformed_depth,       #True
        #         "timeout": timeout,                           # -1
        #         "verbose": verbose,                           
        #     })
        
        """
        parallel_bt: Body Tracking 연산을 병렬로
        undistort_ : 카메라 렌즈 왜곡 보정
        transformed_depth : Depth 를 Color 카메라 로 변환
        """


        self.counter += 1
        skip_count = 0
        while True:
            t0 = time.time()
            res = k4a_module.device_get_capture(timeout)
            self._verify_error(res)
            t1 = time.time()

            if skip_old_atol_ms is None or (t1 - t0) * 1000 > skip_old_atol_ms: # t 설정값보다 짧은경우, 오래된 프레임을 가져온것으로 간주
                break
            else:
                skip_count += 1

        result = dict() # 결과를 담을 딕셔너리
        result['skip_count'] = skip_count # 몇 프레임을 스킵했는지 기록
        if get_color: # True
            if undistort_color: # F  왜곡 보정
                k4a_module.device_get_color_image_undistorted_start()
            else:
                k4a_module.device_get_color_image_start()
        if get_depth: # False
            if undistort_depth:
                if transformed_depth:
                    k4a_module.device_get_transformed_depth_image_undistorted_start()
                else:
                    k4a_module.device_get_depth_image_undistorted_start()
            else:
                if transformed_depth: # True 
                    k4a_module.device_get_transformed_depth_image_start()
                else:
                    raise Exception('get depth_image not implemented')
        if parallel_bt: # T 이미지 처리와 신체 추적 알고리즘 동시
            if get_bt: # T
                if undistort_bt: # F
                    k4a_module.device_get_pose_data_undistorted_start()
                else:
                    k4a_module.device_get_pose_data_start()
            t2 = time.time()

        suff = lambda x: '_undistorted' if x else '' # 결과 딕셔너리 키 이름 뒤에 접미사를 만드는 람다 함수
        if get_depth: # False
            if undistort_depth:
                if transformed_depth: 
                    depth_result = k4a_module.device_get_transformed_depth_image_undistorted_join()
                else:
                    depth_result = k4a_module.device_get_depth_image_undistorted_join()
            else:
                if transformed_depth: # T
                    depth_result = k4a_module.device_get_transformed_depth_image_join()
                else:
                    raise Exception('get depth_image not implemented')
            if depth_result is not None:
                pref = 'tranfromed_' if transformed_depth else ''
                result[f'{pref}depth{suff(undistort_depth)}'] = depth_result
                if get_depth_timestamp:
                    result['depth_timestamp'] = k4a_module.device_get_depth_image_device_timestamp_usec()
            else:
                print('depth_result None')
        if get_color:
            color_result = k4a_module.device_get_color_image_undistorted_join() \
                if undistort_color else k4a_module.device_get_color_image_join()
            if color_result is not None:
                result[f'color{suff(undistort_color)}'] = color_result
                if get_color_timestamp:
                    result['color_timestamp'] = k4a_module.device_get_color_image_device_timestamp_usec()
        if not parallel_bt:
            t2 = time.time()
        if get_bt:
            if not parallel_bt:
                if undistort_bt:
                    k4a_module.device_get_pose_data_undistorted_start()
                else:
                    k4a_module.device_get_pose_data_start()
            pose, body_index_map_result = k4a_module.device_get_pose_data_undistorted_join() \
                if undistort_bt else k4a_module.device_get_pose_data_join()
            if not (pose is None or body_index_map_result is None):
                result['pose'] = pose
                result[f'body_index_map{suff(undistort_bt)}'] = body_index_map_result
        t3 = time.time()

        self.timings['get_capture2_0'].append(t1 - t0) # 캡쳐 대기 시간
        self.timings['get_capture2_1'].append(t2 - t1) # 이미지 처리 시작 ~ BT 시작 사이
        self.timings['get_capture2_2'].append(t3 - t2) # 작업 완료 대기  시간
        self.skip_counts.append(skip_count)

        # 주기 마다 콘솔에 FPS 출력
        if verbose > 0 and self.counter % verbose == 0:
            timings_s = 'k4a timings:\n'
            min_frames = 10
            durations = dict()
            for k, v in sorted(self.timings.items(), key=operator.itemgetter(0)):
                if len(v) > 2 * min_frames:
                    start = int(len(v) / 2)
                    duration = np.array(v)[start:].mean()
                    timings_s += f'\t{k}: {duration * 1000:.3f}ms\n'
                    durations[k] = duration
            if skip_old_atol_ms is not None and len(self.skip_counts) > 2 * min_frames:
                start = int(len(self.skip_counts) / 2)
                timings_s += f'\tskip_count mean: {np.mean(self.skip_counts[start:]):.2f}\n'
            if len(durations) > 0:
                fps = 1 / (durations['get_capture2_0'] + durations['get_capture2_1'] + durations['get_capture2_2'])
                timings_s += f'\tfps: {fps:.1f}\n'

            if len(durations) > 0:
                print(timings_s)
            
        
        return result
    
    # def get_imu_sample(self, timeout=TIMEOUT_WAIT_INFINITE):
    #     """
    #     IMU(가속도계, 자이로스코프) 데이터를 가져옵니다.
    #     Returns:
    #         dict: 'temperature', 'acc_sample', 'acc_timestamp', 'gyro_sample', 'gyro_timestamp'
    #     """
    #     # k4a_module 내부에 해당 바인딩이 되어 있다고 가정합니다.
    #     # 만약 pyk4a 패키지 버전에 따라 이름이 다를 수 있으나, 보통 device_get_imu_sample 입니다.
    #     print(dir(k4a_module))
    #     res, imu_sample = k4a_module.device_get_imu_sample(timeout)
    #     self._verify_error(res)
    #     return imu_sample
    
    def get_imu_rotation_matrix(self):
        """
        C++의 make_R 함수를 Python으로 정확히 이식한 것.
        IMU 가속도 데이터를 이용해 카메라의 기울기를 보정하는 4x4 회전 행렬(_R)을 반환합니다.
        """
        # 1. IMU 샘플 가져오기 (self 이용)
        imu_data = self.get_imu_sample()
        
        # 가속도 데이터 추출 (x, y, z)
        acc = np.array(imu_data['acc_sample'], dtype=np.float32)
        
        # 2. 중력 가속도 벡터 정규화
        acc_norm = np.linalg.norm(acc)
        if acc_norm == 0:
            return np.eye(4) # 에러 방지용 단위 행렬
            
        gravity = acc / acc_norm
        
        # 3. 회전 각도 계산 (C++ imuToRotaionAngle 로직 준수)
        # C++ accPhi: Y, Z 성분 -> Z축 회전에 사용됨
        acc_phi = np.arctan2(gravity[1], gravity[2]) 
        
        # C++ accTheta: X, YZ평면 -> X축 회전에 사용됨
        acc_theta = np.arctan2(gravity[0], np.sqrt(gravity[1]**2 + gravity[2]**2))

        # C++ 코드의 -rotationAngle 반영 (부호 반대)
        theta_z = -acc_phi    # Z축 회전 각도
        theta_x = -acc_theta  # X축 회전 각도

        # 4. 회전 행렬 생성
        # Z축 회전 행렬 (C++ rotZ)
        # [[c, -s, 0], [s, c, 0], ...]
        cz, sz = np.cos(theta_z), np.sin(theta_z)
        Rz = np.array([
            [cz, -sz, 0, 0],
            [sz,  cz, 0, 0],
            [0,   0,  1, 0],
            [0,   0,  0, 1]
        ], dtype=np.float32)
        
        # X축 회전 행렬 (C++ rotX)
        # [[1, 0, 0], [0, c, -s], ...]
        cx, sx = np.cos(theta_x), np.sin(theta_x)
        Rx = np.array([
            [1, 0,   0,  0],
            [0, cx, -sx, 0],
            [0, sx,  cx, 0],
            [0, 0,   0,  1]
        ], dtype=np.float32)
        
        # 5. 좌표축 뒤집기 (Coordinate System Flip)
        # C++: R << -1, 0, 0, 0,  0, -1, 0, 0 ...
        Flip = np.eye(4, dtype=np.float32)
        Flip[0, 0] = -1
        Flip[1, 1] = -1
        
        # 6. 최종 행렬 결합: R = Flip * Rz * Rx
        R_total = Flip @ Rz @ Rx
        
        return R_total


    @property
    def sync_jack_status(self) -> Tuple[bool, bool]:
        res, jack_in, jack_out = k4a_module.device_get_sync_jack()
        self._verify_error(res)
        return jack_in == 1, jack_out == 1

    def _get_color_control(self, cmd: ColorControlCommand) -> Tuple[int, ColorControlMode]:
        res, mode, value = k4a_module.device_get_color_control(cmd)
        self._verify_error(res)
        return value, ColorControlMode(mode)

    def _set_color_control(self, cmd: ColorControlCommand, value: int, mode=ColorControlMode.MANUAL):
        res = k4a_module.device_set_color_control(cmd, mode, value)
        self._verify_error(res)

    @property
    def brightness(self) -> int:
        return self._get_color_control(ColorControlCommand.BRIGHTNESS)[0]

    @property
    def contrast(self) -> int:
        return self._get_color_control(ColorControlCommand.CONTRAST)[0]

    @property
    def saturation(self) -> int:
        return self._get_color_control(ColorControlCommand.SATURATION)[0]

    @property
    def sharpness(self) -> int:
        return self._get_color_control(ColorControlCommand.SHARPNESS)[0]

    @property
    def backlight_compensation(self) -> int:
        return self._get_color_control(ColorControlCommand.BACKLIGHT_COMPENSATION)[0]

    @property
    def gain(self) -> int:
        return self._get_color_control(ColorControlCommand.GAIN)[0]

    @property
    def powerline_frequency(self) -> int:
        return self._get_color_control(ColorControlCommand.POWERLINE_FREQUENCY)[0]

    @property
    def exposure(self) -> int:
        # sets mode to manual
        return self._get_color_control(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE)[0]

    @property
    def exposure_mode_auto(self) -> bool:
        return self._get_color_control(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE)[1] == ColorControlMode.AUTO

    @property
    def whitebalance(self) -> int:
        # sets mode to manual
        return self._get_color_control(ColorControlCommand.WHITEBALANCE)[0]

    @property
    def whitebalance_mode_auto(self) -> bool:
        return self._get_color_control(ColorControlCommand.WHITEBALANCE)[1] == ColorControlMode.AUTO

    @brightness.setter
    def brightness(self, value: int):
        self._set_color_control(ColorControlCommand.BRIGHTNESS, value)

    @contrast.setter
    def contrast(self, value: int):
        self._set_color_control(ColorControlCommand.CONTRAST, value)

    @saturation.setter
    def saturation(self, value: int):
        self._set_color_control(ColorControlCommand.SATURATION, value)

    @sharpness.setter
    def sharpness(self, value: int):
        self._set_color_control(ColorControlCommand.SHARPNESS, value)

    @backlight_compensation.setter
    def backlight_compensation(self, value: int):
        self._set_color_control(ColorControlCommand.BACKLIGHT_COMPENSATION, value)

    @gain.setter
    def gain(self, value: int):
        self._set_color_control(ColorControlCommand.GAIN, value)

    @powerline_frequency.setter
    def powerline_frequency(self, value: int):
        self._set_color_control(ColorControlCommand.POWERLINE_FREQUENCY, value)

    @exposure.setter
    def exposure(self, value: int):
        self._set_color_control(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE, value)

    @exposure_mode_auto.setter
    def exposure_mode_auto(self, mode_auto: bool, value=2500):
        mode = ColorControlMode.AUTO if mode_auto else ColorControlMode.MANUAL
        self._set_color_control(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE, value=value, mode=mode)

    @whitebalance.setter
    def whitebalance(self, value: int, mode=ColorControlMode.MANUAL):
        self._set_color_control(ColorControlCommand.WHITEBALANCE, value)

    @whitebalance_mode_auto.setter
    def whitebalance_mode_auto(self, mode_auto: bool, value=2500):
        mode = ColorControlMode.AUTO if mode_auto else ColorControlMode.MANUAL
        self._set_color_control(ColorControlCommand.WHITEBALANCE, value=value, mode=mode)

    @staticmethod
    def _verify_error(res):
        res = Result(res)
        if res == Result.Failed:
            raise K4AException()
        elif res == Result.Timeout:
            raise K4ATimeoutException()


if __name__ == "__main__":
    k4a = PyK4A(Config())
    k4a.connect()
    print("Connected")
    jack_in, jack_out = k4a.get_sync_jack()
    print("Jack status : in -> {} , out -> {}".format(jack_in, jack_out))
    for _ in range(10):
        color, depth = k4a.device_get_capture(color_only=False)
        print(color.shape, depth.shape)
    k4a.disconnect()
    print("Disconnected")
