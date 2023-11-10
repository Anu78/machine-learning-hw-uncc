#! /opt/homebrew/bin/python3.11 
def canBeIncreasing(nums):
    c = 0
    for i in range(1, len(nums)-1):
        if nums[i] > nums[i-1]:
            continue
        else:
            c += 1 
            if c > 1:
                return False
    
    return True if c <= 1 else False

f = canBeIncreasing([2,3,1,2])
print(f)
