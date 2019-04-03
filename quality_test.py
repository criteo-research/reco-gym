import subprocess
import tempfile
import nbformat
import unittest


class TestNotebookConsistency(unittest.TestCase):
    @staticmethod
    def _execute_notebook(path):
        """
        Execute a Jupyter Notebook from scratch and convert it into another Jupyter Notebook.
           :returns a converted Jupyter Notebook
        """
        with tempfile.NamedTemporaryFile(suffix = ".ipynb") as tmp_notebook:
            args = [
                "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=3600",
                "--output", tmp_notebook.name,
                path
            ]
            subprocess.check_call(args)

            tmp_notebook.seek(0)
            return nbformat.read(tmp_notebook, nbformat.current_nbformat)

    @staticmethod
    def _analise_notebook(notebook):
        """
        Analise notebook cell outputs.

        The function goes through all cell outputs and finds either error or warning.

        :returns a tuple of errors (0th) and warnings (1st)
        """
        errors = []
        warnings = []
        for cell in notebook.cells:
            if 'outputs' in cell:
                for output in cell['outputs']:
                    if output.output_type == "error":
                        errors.append(output)
                    if output.output_type == "warning":
                        warnings.append(output)
        return errors, warnings

    def test_notebooks(self, with_replacement = False):
        """
        Launch Jupyter Notebooks Tests

        This function automates the process of validating of Jupyter Notebooks
        via launching them from scratch and checking that throughout the launching session
        no error/warning occurs.

        Note #1: it is assumed that the current directory
            is the same where a test file is located.
        Note #2: the name of the Notebook should be defined without the extension `*.ipynb'.

            with_replacement: when the flag is set `True' and a Jupyter Notebook
                has successfully passed tests, the Notebook will be replaced
                with a newly generated Notebook with all rendered data, graphs, etc..
        """
        for case in {
            'Getting Started',
            'Compare Agents',
            'Likelihood Agents',
            'Inverse Propensity Score',
            'Explore Exploit Evolution',
            'Complex Time Behaviour',
            'Pure Organic vs Bandit - Number of Online Users',
            'Organic vs Likelihood',
            'IPS vs Non-IPS',
            'Epsilon Worse',
        }:
            with self.subTest(i = case):
                try:
                    notebook = self._execute_notebook(case)
                except Exception:
                    self.fail(f"Case has not passed: {case}")

            errors, warnings = self._analise_notebook(notebook)
            self.assertEqual(errors, [], f"Case '{case}': NOK -- Errors: {errors}")
            self.assertEqual(warnings, [], f"Case '{case}': NOK -- Warnings: {warnings}")

            if with_replacement and len(errors) == 0 and len(warnings) == 0:
                with open(f"{case}.new.ipynb", mode = 'w') as file:
                    file.seek(0)
                    file.write(f"{notebook}")

            print(f"Case '{case}': OK")


if __name__ == '__main__':
    unittest.main()
