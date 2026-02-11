#!/usr/bin/env python3
"""
障碍物加载节点
从 YAML 文件加载静态障碍物并发布到 MoveIt planning scene
"""

import rclpy
from rclpy.node import Node
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
import yaml
from ament_index_python.packages import get_package_share_directory
import os


class ObstacleLoaderNode(Node):
    """从 YAML 文件加载障碍物并发布到 planning scene"""

    def __init__(self):
        super().__init__("obstacle_loader")

        # 声明参数
        self.declare_parameter("obstacles_file", "")
        self.declare_parameter("publish_rate", 1.0)  # Hz

        # 获取参数
        obstacles_file = self.get_parameter("obstacles_file").value
        publish_rate = self.get_parameter("publish_rate").value

        # 加载障碍物数据
        self.obstacles_data = self._load_obstacles_yaml(obstacles_file)

        # 创建 publisher
        self.planning_scene_pub = self.create_publisher(
            PlanningScene, "/planning_scene", 10
        )

        # 创建定时器发布障碍物（确保 move_group 接收到）
        self.timer = self.create_timer(publish_rate, self.publish_obstacles)
        self.publish_count = 0
        self.max_publishes = 3  # 发布几次后停止，确保被接收

        self.get_logger().info("Obstacle loader node started")

    def _load_obstacles_yaml(self, file_path):
        """从YAML文件加载障碍物配置"""
        if not file_path:
            self.get_logger().error("No obstacles file specified!")
            return None

        # 处理相对路径
        if not os.path.isabs(file_path):
            pkg_share = get_package_share_directory("custom_moveit_config")
            file_path = os.path.join(pkg_share, file_path)

        if not os.path.exists(file_path):
            self.get_logger().error(f"Obstacles file not found: {file_path}")
            return None

        try:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
            self.get_logger().info(f"Loaded obstacles from: {file_path}")
            return data
        except Exception as e:
            self.get_logger().error(f"Failed to load obstacles: {e}")
            return None

    def publish_obstacles(self):
        """发布障碍物到 planning scene"""
        if self.publish_count >= self.max_publishes:
            return

        if not self.obstacles_data:
            self.get_logger().warn("No obstacles data to publish")
            return

        # 创建 PlanningScene 消息
        scene_msg = PlanningScene()
        scene_msg.is_diff = True  # 只发布差异（添加的障碍物）

        # 解析每个障碍物
        for obj_data in self.obstacles_data.get("collision_objects", []):
            collision_obj = self._create_collision_object(obj_data)
            if collision_obj:
                scene_msg.world.collision_objects.append(collision_obj)

        # 发布
        self.planning_scene_pub.publish(scene_msg)
        self.publish_count += 1

        self.get_logger().info(
            f"Published {len(scene_msg.world.collision_objects)} obstacles "
            f"({self.publish_count}/{self.max_publishes})"
        )

    def _create_collision_object(self, obj_data):
        """从 YAML 数据创建 CollisionObject"""
        try:
            obj = CollisionObject()
            obj.id = obj_data["id"]
            obj.header.frame_id = obj_data["header"]["frame_id"]
            obj.header.stamp = self.get_clock().now().to_msg()

            # 解析 primitives 和 poses
            for prim_data, pose_data in zip(
                obj_data.get("primitives", []), obj_data.get("primitive_poses", [])
            ):
                # 创建 primitive
                primitive = SolidPrimitive()
                prim_type = prim_data["type"].upper()

                if prim_type == "BOX":
                    primitive.type = SolidPrimitive.BOX
                    primitive.dimensions = prim_data["dimensions"]
                elif prim_type == "SPHERE":
                    primitive.type = SolidPrimitive.SPHERE
                    primitive.dimensions = [prim_data.get("radius", 0.1)]
                elif prim_type == "CYLINDER":
                    primitive.type = SolidPrimitive.CYLINDER
                    primitive.dimensions = [
                        prim_data.get("height", 1.0),
                        prim_data.get("radius", 0.1),
                    ]
                elif prim_type == "CONE":
                    primitive.type = SolidPrimitive.CONE
                    primitive.dimensions = [
                        prim_data.get("height", 1.0),
                        prim_data.get("radius", 0.1),
                    ]
                else:
                    self.get_logger().warn(f"Unknown primitive type: {prim_type}")
                    continue

                obj.primitives.append(primitive)

                # 创建 pose
                pose = Pose()
                pose.position.x = pose_data["position"]["x"]
                pose.position.y = pose_data["position"]["y"]
                pose.position.z = pose_data["position"]["z"]
                pose.orientation.x = pose_data["orientation"]["x"]
                pose.orientation.y = pose_data["orientation"]["y"]
                pose.orientation.z = pose_data["orientation"]["z"]
                pose.orientation.w = pose_data["orientation"]["w"]

                obj.primitive_poses.append(pose)

            # 设置操作类型
            operation = obj_data.get("operation", "ADD").upper()
            if operation == "ADD":
                obj.operation = CollisionObject.ADD
            elif operation == "REMOVE":
                obj.operation = CollisionObject.REMOVE
            elif operation == "MOVE":
                obj.operation = CollisionObject.MOVE
            else:
                obj.operation = CollisionObject.ADD

            return obj

        except Exception as e:
            self.get_logger().error(f"Failed to create collision object: {e}")
            return None


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleLoaderNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
