Yagi Targeting Tool
===================

Not sure where to point your yagi?  Find out.


Installation
------------

Requires:

* Gnuradio
* Gr-gsm


Install using pip:

::

    pip install gr-gsm-yagi-targeting-tool


Use
---

::

    yagi-targeting-tool


You'll be asked to point your antenna in the four cardinal directions (one
at a time).  Detected towers will be presented for each cardinal
direction, but summary output and a recommended direction for pointing
your Yagi antenna for each detected cell tower will be presented after
scanning has completed.

Thanks
------

This is mostly just some extra niceties built atop the built-in
`grgsm_scanner` app in the `gr-gsm` package.  If you find this useful,
keep in mind that the people who contribute to
`gr-gsm <https://github.com/ptrkrysik/gr-gsm>`_ did the actual hard work.

