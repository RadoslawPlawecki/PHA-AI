import sys
import questionary

from phantom.preprocessing.controller import PreprocessingController
# from phantom.cli.features import run_features
# from phantom.cli.classifier import run_classifier

def main():
    print("=== PHANTOM v.0.0.1 ===")
    while True:
        step = questionary.select(
            "Which pipeline step would you like to run?",
            choices=[
                "1) Preprocessing",
                "2) Feature extraction",
                "3) Classifier",
                questionary.Separator(),
                "Exit"
            ]
        ).ask()
        if step is None or step == "Exit":
            print("Exiting pipeline. Goodbye!")
            sys.exit(0)
        if step.startswith("1)"):
            print("\n" + "="*40)
            print("STEP 1: PREPROCESSING")
            print("="*40)
            PreprocessingController().run()
        elif step.startswith("2)"):
            print("\n" + "="*40)
            print("STEP 2: FEATURE EXTRACTION")
            print("="*40)
            print("Module pending implementation...")
            # run_features()
        elif step.startswith("3)"):
            print("\n" + "="*40)
            print("STEP 3: CLASSIFIER")
            print("="*40)
            print("Module pending implementation...")
            # run_classifier()
        print("\n") 

if __name__ == "__main__":
    main()