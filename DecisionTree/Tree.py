
class Tree(object):
    
    def __init__(self,maxDepth):
        self.maxDepth = maxDepth
        self.root = None
        self.is_weights = False
        self.is_randomness = False
        
    def get_root(self,node):
        self.root = node

    

class Node(object):
    
    def __init__(self,val,curDepth):
        self.val = val
        self.curDepth = curDepth
        self.children = {}
        self.label = None
        self.majority = None
        self.purity = False
        self.is_leaf = False
        

    def get_child(self,child_node,attribute):
        self.children[attribute] = child_node

