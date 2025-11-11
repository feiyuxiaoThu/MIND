"""
几何计算工具
"""

import numpy as np
from typing import Tuple, List, Optional


class GeometryUtils:
    """几何计算工具类"""
    
    @staticmethod
    def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """计算欧几里得距离"""
        return np.linalg.norm(p1 - p2)
        
    @staticmethod
    def distance_to_polyline(point: np.ndarray, polyline: np.ndarray) -> float:
        """计算点到折线的最小距离"""
        if len(polyline) < 2:
            return GeometryUtils.euclidean_distance(point, polyline[0])
            
        min_distance = float('inf')
        for i in range(len(polyline) - 1):
            segment_start = polyline[i]
            segment_end = polyline[i + 1]
            
            # 计算点到线段的距离
            distance = GeometryUtils.distance_to_segment(point, segment_start, segment_end)
            min_distance = min(min_distance, distance)
            
        return min_distance
        
    @staticmethod
    def distance_to_segment(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
        """计算点到线段的距离"""
        segment = seg_end - seg_start
        segment_length = np.linalg.norm(segment)
        
        if segment_length == 0:
            return GeometryUtils.euclidean_distance(point, seg_start)
            
        # 计算投影点
        t = max(0, min(1, np.dot(point - seg_start, segment) / (segment_length ** 2)))
        projection = seg_start + t * segment
        
        return GeometryUtils.euclidean_distance(point, projection)
        
    @staticmethod
    def get_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
        """计算两个向量之间的角度"""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)
        
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """将角度归一化到[-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
        
    @staticmethod
    def get_heading_from_velocity(velocity: np.ndarray) -> float:
        """从速度向量计算航向角"""
        return np.arctan2(velocity[1], velocity[0])
        
    @staticmethod
    def rotate_point(point: np.ndarray, angle: float, center: np.ndarray = None) -> np.ndarray:
        """绕中心点旋转点"""
        if center is None:
            center = np.array([0.0, 0.0])
            
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # 旋转矩阵
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])
        
        # 平移到原点，旋转，再平移回去
        translated = point - center
        rotated = rotation_matrix @ translated
        
        return rotated + center
        
    @staticmethod
    def transform_to_local_frame(points: np.ndarray, origin: np.ndarray, 
                               heading: float) -> np.ndarray:
        """将点转换到局部坐标系"""
        # 平移到原点
        translated = points - origin
        
        # 旋转
        cos_heading = np.cos(-heading)
        sin_heading = np.sin(-heading)
        rotation_matrix = np.array([
            [cos_heading, -sin_heading],
            [sin_heading, cos_heading]
        ])
        
        if len(translated.shape) == 1:
            return rotation_matrix @ translated
        else:
            return (rotation_matrix @ translated.T).T
            
    @staticmethod
    def transform_to_global_frame(points: np.ndarray, origin: np.ndarray, 
                               heading: float) -> np.ndarray:
        """将点转换到全局坐标系"""
        # 旋转
        cos_heading = np.cos(heading)
        sin_heading = np.sin(heading)
        rotation_matrix = np.array([
            [cos_heading, -sin_heading],
            [sin_heading, cos_heading]
        ])
        
        if len(points.shape) == 1:
            rotated = rotation_matrix @ points
        else:
            rotated = (rotation_matrix @ points.T).T
            
        # 平移
        return rotated + origin
        
    @staticmethod
    def compute_mahalanobis_distance(point: np.ndarray, mean: np.ndarray, 
                                  covariance: np.ndarray) -> float:
        """计算马氏距离"""
        diff = point - mean
        try:
            inv_covariance = np.linalg.inv(covariance)
            return np.sqrt(diff.T @ inv_covariance @ diff)
        except np.linalg.LinAlgError:
            # 如果协方差矩阵奇异，使用伪逆
            inv_covariance = np.linalg.pinv(covariance)
            return np.sqrt(diff.T @ inv_covariance @ diff)
            
    @staticmethod
    def get_point_mean_distances(points: np.ndarray, mean: np.ndarray) -> np.ndarray:
        """计算点集到均值的距离"""
        if len(points.shape) == 1:
            return np.array([GeometryUtils.euclidean_distance(points, mean)])
        else:
            return np.linalg.norm(points - mean, axis=1)
            
    @staticmethod
    def compute_bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算点集的边界框"""
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        return min_coords, max_coords
        
    @staticmethod
    def compute_convex_hull(points: np.ndarray) -> np.ndarray:
        """计算点集的凸包（简化版，实际应用中可用更高效算法）"""
        # 这里使用简化的凸包算法，实际应用中建议使用scipy.spatial.ConvexHull
        if len(points) < 3:
            return points
            
        # Graham扫描算法的简化版本
        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            
        # 找到最下方的点（y最小，如果相同则x最小）
        start_idx = np.argmin(points[:, 1])
        if np.sum(points[:, 1] == points[start_idx, 1]) > 1:
            same_y = np.where(points[:, 1] == points[start_idx, 1])[0]
            start_idx = same_y[np.argmin(points[same_y, 0])]
            
        hull = [points[start_idx]]
        current = points[start_idx]
        
        while True:
            next_point = None
            for candidate in points:
                if np.array_equal(candidate, current):
                    continue
                    
                if next_point is None:
                    next_point = candidate
                    continue
                    
                # 检查candidate是否在current-next_point的左侧
                cross = cross_product(current, next_point, candidate)
                if cross < 0:
                    next_point = candidate
                elif cross == 0:
                    # 如果共线，选择更远的点
                    if GeometryUtils.euclidean_distance(current, candidate) > \
                       GeometryUtils.euclidean_distance(current, next_point):
                        next_point = candidate
                        
            if next_point is None or np.array_equal(next_point, hull[0]):
                break
                
            hull.append(next_point)
            current = next_point
            
        return np.array(hull)
        
    @staticmethod
    def check_line_intersection(p1: np.ndarray, p2: np.ndarray, 
                              p3: np.ndarray, p4: np.ndarray) -> bool:
        """检查两条线段是否相交"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
            
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
        
    @staticmethod
    def interpolate_polyline(polyline: np.ndarray, num_points: int) -> np.ndarray:
        """在折线上插值生成指定数量的点"""
        if len(polyline) < 2:
            return polyline
            
        if num_points <= len(polyline):
            return polyline[:num_points]
            
        # 计算每段的长度
        segment_lengths = []
        total_length = 0
        for i in range(len(polyline) - 1):
            length = GeometryUtils.euclidean_distance(polyline[i], polyline[i + 1])
            segment_lengths.append(length)
            total_length += length
            
        # 均匀采样
        interpolated_points = [polyline[0]]
        current_length = 0
        segment_idx = 0
        segment_start = polyline[0]
        
        for i in range(1, num_points - 1):
            target_length = i * total_length / (num_points - 1)
            
            # 找到当前长度所在的段
            while segment_idx < len(segment_lengths) and \
                  current_length + segment_lengths[segment_idx] < target_length:
                current_length += segment_lengths[segment_idx]
                segment_idx += 1
                segment_start = polyline[segment_idx]
                
            if segment_idx >= len(segment_lengths):
                break
                
            # 在当前段内插值
            remaining_length = target_length - current_length
            t = remaining_length / segment_lengths[segment_idx]
            segment_end = polyline[segment_idx + 1]
            
            interpolated_point = segment_start + t * (segment_end - segment_start)
            interpolated_points.append(interpolated_point)
            
        interpolated_points.append(polyline[-1])
        
        return np.array(interpolated_points)