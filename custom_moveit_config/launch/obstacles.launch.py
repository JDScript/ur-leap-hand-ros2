import rclpy
from rclpy.node import Node
from moveit_msgs.msg import PlanningScene, CollisionObject
import yaml
from geometry_msgs.msg import Pose

class ObstacleLoader(Node):
    def __init__(self):
        super().__init__('obstacle_loader')
        self.pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        self.timer = self.create_timer(1.0, self.load_obstacles)
        self.loaded = False

    def load_obstacles(self):
        if self.loaded:
            return
        with open('/path/to/obstacles.yaml', 'r') as f:
            data = yaml.safe_load(f)
        scene = PlanningScene()
        scene.is_diff = True
        for obj_data in data['collision_objects']:
            obj = CollisionObject()
            obj.id = obj_data['id']
            obj.header.frame_id = obj_data['header']['frame_id']
            # 这里只是 BOX 的例子，可以根据 primitives 类型扩展
            for prim, pose in zip(obj_data['primitives'], obj_data['primitive_poses']):
                from shape_msgs.msg import SolidPrimitive
                prim_msg = SolidPrimitive()
                if prim['type'] == 'BOX':
                    prim_msg.type = SolidPrimitive.BOX
                    prim_msg.dimensions = prim['dimensions']
                obj.primitives.append(prim_msg)
                pose_msg = Pose()
                pose_msg.position.x = pose['position']['x']
                pose_msg.position.y = pose['position']['y']
                pose_msg.position.z = pose['position']['z']
                pose_msg.orientation.x = pose['orientation']['x']
                pose_msg.orientation.y = pose['orientation']['y']
                pose_msg.orientation.z = pose['orientation']['z']
                pose_msg.orientation.w = pose['orientation']['w']
                obj.primitive_poses.append(pose_msg)
            obj.operation = CollisionObject.ADD
            scene.world.collision_objects.append(obj)
        self.pub.publish(scene)
        self.get_logger().info("Obstacles loaded")
        self.loaded = True

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleLoader()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()