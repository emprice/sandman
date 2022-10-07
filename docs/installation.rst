Downloading and installing
==========================

The full source of :rainbow:`sandman` can be downloaded from
`its GitHub repository <https://github.com/emprice/sandman>`_. If you want
to change the code and possibly submit an issue or pull request, you can also
clone the repository with the following command.

.. code-block:: none

   git clone https://github.com/emprice/sandman.git

To install the package, you can use :code:`pip`.

.. note::

   :rainbow:`sandman` is not currently on PyPI, and, unless anyone actually uses
   it, there really isn't any reason to submit it there. If you feel strongly
   that you would like to see this package on PyPI, please let me know!

To install from local files (cloned or downloaded from GitHub):

.. code-block:: none

   pip3 install [--force-reinstall --ignore-installed] [--user] $PWD

To install from GitHub without downloading or cloning directly:

.. code-block:: none

   pip3 install [--force-reinstall --ignore-installed] [--user] git+https://github.com/emprice/sandman.git@main

The options in brackets are not strictly necessary to install
:rainbow:`sandman`, but they can be useful.

.. option:: --force-reinstall

   When working with local files especially, :code:`pip` does not want to
   install a package with the same version as the one already installed.
   This option forces a fresh install without a version bump.

.. option:: --ignore-installed

   Unfortunately, if you use :code:`--force-reinstall` by itself, :code:`pip`
   tends to try to reinstall the entire dependency tree of the package you
   want to update. This option tells :code:`pip` to *only* reinstall the
   package you ask for.

.. option:: --user

   If you work on a system without root privileges, you *may* need to use
   this option to install the package for your user. However, rather than
   changing the system Python installation, it is generally recommended to
   work in a virtual environment, where you don't need root privileges
   to install new packages and cannot break the system Python by accident.
