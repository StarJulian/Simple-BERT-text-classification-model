class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        WHITE, GRAY = 0, 1
        res = []
        stack = [(WHITE, root)]
        while stack:
            color, node = stack.pop()
            if node is None: continue
            if color == WHITE:
                stack.append((WHITE, node.right))
                stack.append((GRAY, node))
                stack.append((WHITE, node.left))
            else:
                res.append(node.val)
        return res


class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        WHITE, GRAY = 0, 1
        res = []
        stack = [(WHITE, root)]

        while stack:
            color, node = stack.pop()
            if node is None:
                continue

            if color == WHITE:
                stack.append((GRAY, node))
                if node.right:
                    stack.append((WHITE, node.right))
                if node.left:
                    stack.append((WHITE, node.left))
            else:
                res.append(node.val)

        return res


class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        WHITE, GRAY = 0, 1
        res = []
        stack = [(WHITE, root)]

        while stack:
            color, node = stack.pop()
            if node is None:
                continue

            if color == WHITE:
                if node.right:
                    stack.append((WHITE, node.right))
                stack.append((GRAY, node))
                if node.left:
                    stack.append((WHITE, node.left))
            else:
                res.append(node.val)

        return res


















