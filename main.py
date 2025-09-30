from src.queries.query_executor import QueryExecutor
from src.trivial_solutions.iterative_neural_network import IterativeNN

def alarm_example():
    sdd_file = "examples/alarm/alarm_balanced.sdd"
    json_file = "examples/alarm/alarm.json"
    query_executor = QueryExecutor(sdd_file, json_file, IterativeNN)
    conditional_prob = query_executor.execute_query([1], [5])
    print(f"P(burglary | calls(john)) = {conditional_prob:.3f}\n")
    conditional_prob = query_executor.execute_query([1, 2, 3, 4, 5], [])
    print(f"P(burglary, earthquake, hears_alarm, alarm, calls) = {conditional_prob:.3f}\n")

def main():
    print("\n======================== alarm example ========================\n")
    alarm_example()

if __name__ == "__main__":
    main()
