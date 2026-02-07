from spill_queue import SpillQueue

spill_queue = SpillQueue(2)

for i in range(5):
    spill_queue.put(2, i, "test_payload")

# for i in range(5):
#     print(spill_queue.pop_next(1))
print(spill_queue.pop_exact(2,3))

