import json
import copy
import re


def augment_dataset(input_file='simplified_tests.json', output_file='augmented_tests.json'):
    print(f"--- Augmenting {input_file} ---")

    with open(input_file, 'r', encoding='utf-8') as f:
        original_tests = json.load(f)

    new_tests = []

    # Track entities to build chains later (e.g., "User": {"create": test, "delete": test})
    entity_map = {}

    print(f"Original Count: {len(original_tests)}")

    for test in original_tests:
        # Keep the original
        new_tests.append(test)

        # Extract basic info
        steps = test.get('steps', [])
        if not steps: continue

        first_action = steps[0].get('action', '').lower()
        test_name = test.get('test_name', '')

        # --- STRATEGY 1: NEGATIVE TEST GENERATION ---
        # If it's a data entry operation (POST/PUT/Create), generate failure scenarios
        if "post" in first_action or "create" in first_action or "put" in first_action:
            # Variant A: Empty/Missing Data
            neg_test = copy.deepcopy(test)
            neg_test['test_name'] = f"Negative: {test_name} - Empty Payload"
            neg_test['objective'] = "Verify system handles empty input gracefully"
            neg_test['steps'][0]['data'] = "{}"  # Empty JSON
            neg_test['steps'][0]['expected_result'] = "400 Bad Request or Validation Error"
            new_tests.append(neg_test)

            # Variant B: Invalid Data Types
            inv_test = copy.deepcopy(test)
            inv_test['test_name'] = f"Negative: {test_name} - Invalid Data Types"
            inv_test['objective'] = "Verify system rejects invalid field types (e.g. string for int)"
            inv_test['steps'][0]['data'] = "{\"id\": \"invalid_string\", \"age\": -5}"
            inv_test['steps'][0]['expected_result'] = "400 Bad Request"
            new_tests.append(inv_test)

        # --- STRATEGY 2: SECURITY INJECTION (Fuzzing) ---
        # Add basic security checks for any data-heavy request
        if "data" in steps[0] and len(steps[0]['data']) > 5:
            sec_test = copy.deepcopy(test)
            sec_test['test_name'] = f"Security: {test_name} - SQL Injection"
            sec_test['objective'] = "Verify API is resilient to SQL injection attempts"
            sec_test['steps'][0]['data'] = "' OR '1'='1"
            sec_test['steps'][0]['expected_result'] = "400 Error or 200 (Safe Empty List). MUST NOT expose DB error."
            new_tests.append(sec_test)

        # --- STRATEGY 3: ENTITY MAPPING FOR CHAINS ---
        # Try to guess the entity name (e.g., "User entity-Create" -> "User")
        # This regex looks for words before "entity" or "test"
        match = re.search(r"([a-zA-Z]+) (entity|test)", test_name, re.IGNORECASE)
        if match:
            entity = match.group(1).lower()
            if entity not in entity_map: entity_map[entity] = {}

            if "create" in test_name.lower():
                entity_map[entity]['create'] = test
            elif "delete" in test_name.lower():
                entity_map[entity]['delete'] = test
            elif "get" in test_name.lower():
                entity_map[entity]['get'] = test

    # --- STRATEGY 4: WORKFLOW GENERATION (Chaining) ---
    # Combine Create -> Get -> Delete into one test case
    print("\nGenerating Workflow Chains...")
    for entity, actions in entity_map.items():
        if 'create' in actions and 'delete' in actions:
            chain_test = {
                "test_name": f"Lifecycle: {entity.capitalize()} Full Cycle",
                "objective": f"Verify full lifecycle of {entity}: Create, Verify, and Delete",
                "preconditions": "Database valid",
                "steps": []
            }

            # Step 1: Create
            s1 = copy.deepcopy(actions['create']['steps'][0])
            s1['position'] = 1
            chain_test['steps'].append(s1)

            # Step 2: Get (if available)
            if 'get' in actions:
                s2 = copy.deepcopy(actions['get']['steps'][0])
                s2['position'] = 2
                s2['action'] = f"{s2['action']} (Using ID from Step 1)"
                chain_test['steps'].append(s2)

            # Step 3: Delete
            s3 = copy.deepcopy(actions['delete']['steps'][0])
            s3['position'] = 3 if 'get' in actions else 2
            s3['action'] = f"{s3['action']} (Using ID from Step 1)"
            chain_test['steps'].append(s3)

            new_tests.append(chain_test)
            print(f"  + Generated chain for '{entity}'")

    # Save Results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_tests, f, indent=2)

    print(f"\nSuccess! Augmented dataset saved to {output_file}")
    print(f"New Total Count: {len(new_tests)} (Was: {len(original_tests)})")


if __name__ == "__main__":
    augment_dataset()