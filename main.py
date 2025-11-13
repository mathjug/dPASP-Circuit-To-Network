from src.queries.query_executor import QueryExecutor
from src.network_converter.iterative_neural_network import IterativeNN
from src.optimizer.dataset_generator import AlarmDatasetGenerator
from src.optimizer.probability_optimizer import ProbabilityOptimizer

sdd_file = "examples/alarm/alarm_balanced.sdd"
json_file = "examples/alarm/alarm.json"

def alarm_inference_example():
    query_executor = QueryExecutor(sdd_file, json_file, IterativeNN)
    conditional_prob = query_executor.execute_query([1], [5])
    print(f"P(burglary | calls(john)) = {conditional_prob:.3f}\n")
    conditional_prob = query_executor.execute_query([1, 2, 3, 4, 5], [])
    print(f"P(burglary, earthquake, hears_alarm, alarm, calls) = {conditional_prob:.3f}")

def alarm_optimization_example():
    neural_network = IterativeNN(sdd_file, json_file)
    X_train = AlarmDatasetGenerator(neural_network).generate_dataset(num_samples=50000)
    probability_optimizer = ProbabilityOptimizer(neural_network)
    literals_to_learn = [1, 2, 3]
    final_learned_params, final_loss = probability_optimizer.learn_probability(literals_to_learn, X_train, learning_rate=0.1, num_epochs=5000)
    print(f"Final loss: {final_loss:.3f}\n")
    for literal in literals_to_learn:
        print(f"Final learned probability of X_{literal}: {final_learned_params[literal]:.3f}")

def main():
    print("\n======================== alarm inference example ========================\n")
    alarm_inference_example()
    print("\n======================== alarm optimization example ========================\n")
    alarm_optimization_example()

if __name__ == "__main__":
    main()
