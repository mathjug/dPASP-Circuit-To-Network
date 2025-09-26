from src.queries.query_executor import QueryExecutor

def alarm_example():
    sdd_file = "examples/alarm/alarm_balanced.sdd"
    json_file = "examples/alarm/alarm.json"
    vtree_file = "examples/alarm/alarm_balanced.vtree"
    query_executor = QueryExecutor(sdd_file, json_file, vtree_file)
    conditional_prob = query_executor.execute_query(5, [4])
    print(f"\nP(calls(john) | alarm) = {conditional_prob:.3f}\n")

def main():
    print("\n======================== alarm example ========================\n")
    alarm_example()

if __name__ == "__main__":
    main()
