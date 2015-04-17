from cx_Freeze import setup, Executable

# On appelle la fonction setup
setup(
    name = "votre_programme",
    version = "1.0",
    description = "Votre programme",
    executables = [Executable("HARDIPrep.py")],
)