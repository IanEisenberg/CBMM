# psiturk_tutorial
Tutorial for learning psiTurk for running experiments on Mechanical Turk

For CBMM Summer School 2017

## psiTurk installation

If you use Windows, you are out of luck -- psiTurk only works on Unix systems due to the limitations of Windows (though in the future they are aiming to put the system on the cloud so everyone can use it).

If you use a Mac or Linux, you should first install [python 2.7](https://www.python.org/downloads/release/python-2713/), and [pip](https://pip.pypa.io/en/stable/installing/) (which should be automatically installed unless you have an older python version).

If you are using a Mac, you may need to install a C compiler. The easiest way to do this is to install XCode from the App Store.

Next, all you need to do is open a terminal and enter:

```
pip install psiturk
```

If that doesn't work, you may want to try it with `sudo`:

```
sudo pip install psiturk
```

If you want to actually use psiTurk for posting experiments to Mechanical Turk, you will need to make both a [Mechanical Turk account](http://psiturk.readthedocs.io/en/latest/amt_setup.html) and a [psiTurk account](http://psiturk.readthedocs.io/en/latest/psiturk_org_setup.html). These won't be necessary to play with psiTurk in the tutorial, but would be needed if you hope to use this for your experiments -- however, because you will need to enter a credit card for a Mechanical Turk account, this is only suggested if you plan on actually using this in the future.

Finally, to test that this has installed correctly, simply use the terminal and navigate to the `cbmm-psiturk-exercise`, then just enter the command `psiturk`. If it starts up correctly (and you should be able to tell by the giant psiTurk ASCII that will appear), you can just enter the `exit` command to leave the psiTurk system.
