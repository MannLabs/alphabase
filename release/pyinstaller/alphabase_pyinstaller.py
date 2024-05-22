if __name__ == "__main__":
    try:
        import multiprocessing

        import alphabase.gui

        multiprocessing.freeze_support()
        alphabase.gui.run()
    except Exception:
        import sys
        import traceback

        exc_info = sys.exc_info()
        # Display the *original* exception
        traceback.print_exception(*exc_info)
        input("Something went wrong, press any key to continue...")
