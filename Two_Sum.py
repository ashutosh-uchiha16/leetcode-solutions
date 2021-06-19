def towSum(self, nums, target):
    a = {}
        for i, v in enumerate(nums):
            targ = target - v
            if targ in a:
                return [a[targ], i]
            a[v] = i