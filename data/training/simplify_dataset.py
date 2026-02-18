import json


def simplify_qase_export(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    simplified_tests = []

    def process_suites(suite_list):
        for suite in suite_list:
            # 1. Process all cases in the current suite
            for case in suite.get('cases', []):
                # Build a clean dictionary with only essential info
                test_info = {
                    "test_name": case.get('title', 'Unknown Title'),
                    "objective": case.get('description', ''),
                    "preconditions": case.get('preconditions', ''),
                    "steps": []
                }

                # Extract step details
                for step in case.get('steps', []):
                    step_info = {
                        "action": step.get('action', ''),
                        "expected_result": step.get('expected_result', ''),
                        "data": step.get('data', '')  # Keep data if relevant (e.g., input params)
                    }
                    test_info['steps'].append(step_info)

                simplified_tests.append(test_info)

            # 2. Recursively process any nested suites
            if 'suites' in suite and suite['suites']:
                process_suites(suite['suites'])

    # Start processing from the root
    if 'suites' in data:
        process_suites(data['suites'])

    # Save the simplified list
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(simplified_tests, f, indent=2, ensure_ascii=False)

    print(f" Successfully extracted {len(simplified_tests)} tests to '{output_file}'")


# usage
simplify_qase_export('QA-Backend-Data.json', 'simplified_tests.json')