import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.Policy2352294_2352554_2352520_2352100_2352002 import Policy2352294_2352554_2352520_2352100_2352002
import time

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 100

if __name__ == "__main__":
    # # Reset the environment
    # observation, info = env.reset(seed=42)

    # # Test GreedyPolicy
    # gd_policy = GreedyPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = gd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         print(info)
    #         observation, info = env.reset(seed=ep)
    #         ep += 1

    # Reset the environment
    # observation, info = env.reset(seed=42)

    # Test Policy2352294
    # rd_policy = Policy2352294()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = rd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         print(info)
    #         observation, info = env.reset(seed=ep)
    #         ep += 1

    # Uncomment the following code to test your policy
    # # Reset the environment
    # observation, info = env.reset(seed=42)
    # # print(info)

    # Policy2352294_2352554_2352520_2352100_2352002 = Policy2352294_2352554_2352520_2352100_2352002(policy_id= 1 )
    # for _ in range(200):
    #     action = Policy2352294_2352554_2352520_2352100_2352002.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     print(info)

    #     if terminated or truncated:
    #         observation, info = env.reset()

    observation, info = env.reset(seed=42)
    print(info)
    list_prods = observation["products"]
    total_quantity = sum(prod["quantity"] for prod in list_prods)
    print("Num of stocks: " + str(total_quantity))

    Policy2352294_2352554_2352520_2352100_2352002 = Policy2352294_2352554_2352520_2352100_2352002(policy_id= 2 )
    total_action_time = 0  # Khởi tạo biến để lưu tổng thời gian cho tất cả các hành động
    i = 1
    for _ in range(200):
        start_time = time.time()  # Bắt đầu thời gian cho mỗi hành động
        action = Policy2352294_2352554_2352520_2352100_2352002.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        print("Action " + str(i) + " : " + str(info))
        i = i + 1

        if terminated or truncated:
            observation, info = env.reset()

        action_time = time.time() - start_time  # Tính thời gian cho hành động hiện tại
        total_action_time += action_time  # Cộng dồn vào tổng thời gian
        print("Action " + str(i) + " completed in {:.2f} seconds".format(action_time))

    print("Total action time: {:.2f} seconds".format(total_action_time))  # In tổng thời gian cho tất cả các hành động

    env.close()
