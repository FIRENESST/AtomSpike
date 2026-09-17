from multiprocessing import freeze_support

from atomspike.cli import main

if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
