import sys

def log_StanceEval(Ex_name,logger,val_or_test):
    folder_path ='-'.join(("checkpoints/"+Ex_name).split("-")[:-1]) 
    fn_guess = folder_path+ "/"+val_or_test+"_pred_File.txt"
    fn_gold = folder_path+ "/"+val_or_test+"_gt_file.txt"

    # logger.experiment.log_metric("val_or_test",val_or_test)
    try:
        with open(fn_gold, 'r') as f_gold, open(fn_guess, 'r') as f_guess:
            gold_lines = f_gold.readlines()
            guess_lines = f_guess.readlines()
    except IOError as e:
        sys.stderr.write(f"Error: Cannot open file: {e.filename}\n")
        sys.exit(1)

    gold_lines = [line.strip() for line in gold_lines]
    guess_lines = [line.strip() for line in guess_lines]

    if len(guess_lines) != len(gold_lines):
        sys.stderr.write("\nError: Make sure the number of lines in your prediction file is same as that in the gold-standard file!\n")
        sys.stderr.write(f"The gold-standard file contains {len(gold_lines)} lines, but the prediction file contains {len(guess_lines)} lines.\n")
        sys.exit(1)

    targets = ["Women empowerment", "Covid Vaccine", "Digital Transformation"]
    cats = ["FAVOR", "AGAINST", "NONE"] 

    # Initialize dictionaries to store statistics for each target
    num_of_true_pos_of_each_target = {target: {cat: 0 for cat in cats} for target in targets}
    num_of_guess_of_each_target = {target: {cat: 0 for cat in cats} for target in targets}
    num_of_gold_of_each_target = {target: {cat: 0 for cat in cats} for target in targets}

    for gold_line, guess_line in zip(gold_lines, guess_lines):
        if gold_line == "ID\tTarget\tTweet\tStance":
            continue

        gold_arr = gold_line.split("\t")
        guess_arr = guess_line.split("\t")

        if len(gold_arr) != 4:
            print(f"\nError: the following line in the gold-standard file does not have a correct format:\n\n{gold_line}\n\n")
            print("Correct format: ID<Tab>Target<Tab>Tweet<Tab>Stance\n")
            sys.exit(1)

        if len(guess_arr) != 4:
            print(f"\nError: the following line in your prediction file does not have a correct format:\n\n{guess_line}\n\n")
            print("Correct format: ID<Tab>Target<Tab>Tweet<Tab>Stance\n")
            sys.exit(1)

        gold_target = gold_arr[1]
        gold_lbl = gold_arr[3]
        guess_target = guess_arr[1]
        guess_lbl = guess_arr[3]

        if gold_target not in targets:
            print(f"\nError: the target \"{gold_target}\" in the gold-standard file is invalid:\n\n{gold_line}\n\n")
            sys.exit(1)

        if guess_target not in targets:
            print(f"\nError: the target \"{guess_target}\" in the prediction file is invalid:\n\n{guess_line}\n\n")
            sys.exit(1)

        if gold_lbl not in cats:
            print(f"\nError: the stance label \"{gold_lbl}\" in the gold-standard file is invalid:\n\n{gold_line}\n\n")
            sys.exit(1)

        if guess_lbl not in cats:
            print(f"\nError: the stance label \"{guess_lbl}\" in the prediction file is invalid:\n\n{guess_line}\n\n")
            sys.exit(1)

        num_of_gold_of_each_target[gold_target][gold_lbl] += 1
        num_of_guess_of_each_target[guess_target][guess_lbl] += 1

        if guess_lbl == gold_lbl:
            num_of_true_pos_of_each_target[guess_target][guess_lbl] += 1

    prec_by_target = {target: {cat: 0 for cat in cats} for target in targets}
    recall_by_target = {target: {cat: 0 for cat in cats} for target in targets}
    f_by_target = {target: {cat: 0 for cat in cats} for target in targets}
    macro_f_by_target = {target: 0.0 for target in targets}

    for target in targets:
        macro_f = 0.0
        n_cat = 0

        for cat in cats:
            n_tp = num_of_true_pos_of_each_target[target][cat]
            n_guess = num_of_guess_of_each_target[target][cat]
            n_gold = num_of_gold_of_each_target[target][cat]

            p = 0
            r = 0
            f = 0

            if n_guess != 0:
                p = n_tp / n_guess
            if n_gold != 0:
                r = n_tp / n_gold
            if p + r != 0:
                f = 2 * p * r / (p + r)

            prec_by_target[target][cat] = p
            recall_by_target[target][cat] = r
            f_by_target[target][cat] = f

            if cat in ["FAVOR", "AGAINST"]:
                n_cat += 1
                macro_f += f

        macro_f = macro_f / n_cat
        macro_f_by_target[target] = macro_f

        # Print results for each target
        for cat in cats:
            if cat in ["FAVOR", "AGAINST"]:
                logger.experiment.log_metric(f"{val_or_test} {target} {cat:<9} precision", prec_by_target[target][cat])
                logger.experiment.log_metric(f"{val_or_test} {target} {cat:<9} recall", recall_by_target[target][cat])
                logger.experiment.log_metric(f"{val_or_test} {target} {cat:<9} f-score",f_by_target[target][cat])
        logger.experiment.log_metric(f"{val_or_test}  {target} Macro F", macro_f)

    # Compute overall macro F1-score across all targets
    overall_macro_f = sum(macro_f_by_target.values()) / len(targets)
    logger.experiment.log_metric(f"Overall Macro F1-score across all targets {val_or_test}",overall_macro_f)
