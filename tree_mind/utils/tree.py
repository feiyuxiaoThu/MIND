"""
通用树结构实现
"""

from typing import Dict, List, Optional, Any


class Node:
    """树节点基类"""
    
    def __init__(self, key: str, parent_key: Optional[str], data: Any):
        self.key = key
        self.parent_key = parent_key
        self.data = data
        self.children_keys: List[str] = []
        
    def add_child(self, child_key: str):
        """添加子节点"""
        if child_key not in self.children_keys:
            self.children_keys.append(child_key)
            
    def remove_child(self, child_key: str):
        """移除子节点"""
        if child_key in self.children_keys:
            self.children_keys.remove(child_key)
            
    def is_leaf(self) -> bool:
        """判断是否为叶节点"""
        return len(self.children_keys) == 0
        
    def is_root(self) -> bool:
        """判断是否为根节点"""
        return self.parent_key is None


class Tree:
    """树结构"""
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        
    def add_node(self, node: Node):
        """添加节点"""
        self.nodes[node.key] = node
        
        # 更新父节点的子节点列表
        if node.parent_key is not None and node.parent_key in self.nodes:
            self.nodes[node.parent_key].add_child(node.key)
            
    def remove_node(self, key: str):
        """移除节点及其所有子节点"""
        if key not in self.nodes:
            return
            
        # 递归移除所有子节点
        node = self.nodes[key]
        for child_key in list(node.children_keys):
            self.remove_node(child_key)
            
        # 从父节点的子节点列表中移除
        if node.parent_key is not None and node.parent_key in self.nodes:
            self.nodes[node.parent_key].remove_child(key)
            
        # 移除节点本身
        del self.nodes[key]
        
    def get_node(self, key: str) -> Optional[Node]:
        """获取节点"""
        return self.nodes.get(key)
        
    def get_root(self) -> Optional[Node]:
        """获取根节点"""
        for node in self.nodes.values():
            if node.is_root():
                return node
        return None
        
    def get_leaf_nodes(self) -> List[Node]:
        """获取所有叶节点"""
        leaf_nodes = []
        for node in self.nodes.values():
            if node.is_leaf():
                leaf_nodes.append(node)
        return leaf_nodes
        
    def get_children(self, key: str) -> List[Node]:
        """获取子节点"""
        if key not in self.nodes:
            return []
            
        node = self.nodes[key]
        children = []
        for child_key in node.children_keys:
            if child_key in self.nodes:
                children.append(self.nodes[child_key])
        return children
        
    def get_parent(self, key: str) -> Optional[Node]:
        """获取父节点"""
        if key not in self.nodes:
            return None
            
        node = self.nodes[key]
        if node.parent_key is None:
            return None
            
        return self.nodes.get(node.parent_key)
        
    def get_path_to_root(self, key: str) -> List[Node]:
        """获取从节点到根节点的路径"""
        path = []
        current_key = key
        
        while current_key is not None:
            if current_key in self.nodes:
                path.append(self.nodes[current_key])
                current_key = self.nodes[current_key].parent_key
            else:
                break
                
        return path
        
    def size(self) -> int:
        """获取树的大小"""
        return len(self.nodes)
        
    def depth(self) -> int:
        """获取树的深度"""
        if not self.nodes:
            return 0
            
        max_depth = 0
        for node in self.nodes.values():
            depth = len(self.get_path_to_root(node.key)) - 1
            max_depth = max(max_depth, depth)
            
        return max_depth
        
    def is_empty(self) -> bool:
        """判断树是否为空"""
        return len(self.nodes) == 0