Project motivation
==================

Making scientific data and visualizations accessible to as many people as
possible is clearly an excellent goal. However, everyone has different
aesthetic preferences, and sacrificing visual appeal for accessibility is
not generally necessary. If you have found that you are unsatisfied or bored
with existing perceptually uniform colormaps, this project may be what you need.

Color perception differs between individuals
--------------------------------------------

Red-green colorblindness is a common form of color vision deficiency (CVD),
but it is not the only one. Problems with one or more optical cones in the
human eye can lead to very different perceptions of the same image.
:ref:`The table below <cvd>` gives a rough idea of how common ---
or uncommon --- different kinds of CVD are today.

It is not sufficient, then, to mash a few colors together and call the result a
"good" colormap. Instead, we want to ensure that the eye perceives a linearly
varying intensity along the colormap, independent of hue. The colors
themselves become an aesthetic element that is not necessary to understand
the data being conveyed.

As a side note, perceptually uniform colormaps should also produce images
that can be understood in grayscale, which is helpful if an image needs to
used in print media without color.

.. caution::

   :ref:`This table <cvd>` uses unverified data from Wikipedia. It is, at best,
   a rough guideline for illustrative purposes *only*. This data is not mine,
   and you should not cite it from this page (or, indeed, from Wikipedia,
   as citations were not listed at time of writing). If you require quantitative
   data on color vision deficiency, please consult a medical journal.

.. list-table:: Approximate rates of CVD in the population, split by genetic sex `[source] <https://en.wikipedia.org/wiki/Color_blindness#Epidemiology>`_
   :name: cvd
   :width: 80%
   :header-rows: 1
   :stub-columns: 1

   * -
     - Male
     - Female
   * - Protanomaly
     - 1.3%
     - 0.02%
   * - Deuteranomaly
     - 5.0%
     - 0.35%
   * - Tritanomaly
     - 0.0001%
     - 0.0001%

Well-known perceptually uniform colormaps
-----------------------------------------

If you use :mod:`matplotlib` at all, you are undoubtedly familiar with its
perceptually uniform colormaps: :colormap:`matplotlib.cm.viridis`,
:colormap:`matplotlib.cm.plasma`, :colormap:`matplotlib.cm.inferno`,
:colormap:`matplotlib.cm.magma`, and :colormap:`matplotlib.cm.cividis`.
And, to be clear, *there's nothing wrong with these colormaps!* Personally,
I like *magma*, but the lightest color is sometimes too light for a particular
application; similarly, I like *plasma*, but the bright yellow looks bad on
a white background you might use in a journal article figure. *plasma*,
*inferno*, and *magma* also all use similar base colors, making them less
distinct than, say, *viridis* compared to *plasma*.

Another good source of pre-designed colormaps is the
:any:`cmasher <cmasher:cmr_cmaps>` library. It includes some maps that I
really like, such as :colormap:`cmasher.ember`, :colormap:`cmasher.bubblegum`,
and, of course :colormap:`cmasher.pride`. Still, for a particular project,
perhaps I need or want something a bit different.

What :rainbow:`sandman` can do
------------------------------

Simply put, :rainbow:`sandman` will take any colors you dream up and try to make
them into a "nice" sequential, diverging, or cyclic colormap. Your imagination
(and, perhaps, number of available CPU cores) is the limit! In the end, you will
have a JSON file with lots of new colormaps, a score that indicates how close to
or far from perceptually uniform that colormap is, and, just for fun, a
pseudo-random, pronounceable name for that colormap based on the input
parameters that generated it. :rainbow:`sandman` can also plot the "best"
colormaps from one of these JSON blobs so you can choose the one(s) you like
best.

What :rainbow:`sandman` cannot do
---------------------------------

While :rainbow:`sandman` assigns a numeric score to the colormaps it generates
to give the user a rough idea of how well the optimization step worked, this
is relative and somewhat subjective: You could easily choose a different
optimization objective function and get a different result. So these numbers
are useful but should not be treated as absolute quality measures.

Furthermore, :rainbow:`sandman` attempts to simulate human color perception
using the :code:`CAM02-UCS` model from
:any:`colorspacious <colorspacious:overview>`, but, since this is just a model,
it is probably not perfect, and it could be revised in the future. Treat it as
you would any other model.
