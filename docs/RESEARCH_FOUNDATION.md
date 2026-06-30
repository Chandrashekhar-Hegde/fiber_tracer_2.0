# Research Foundation

A literature survey establishing the academic and research background for Fiber
Tracer, spanning the analysis → prediction → digital-twin continuum for X-ray
computed tomography (XCT) of fiber-reinforced composites, framed for
high-reliability (aerospace/defense) use. References are numbered and listed in
section 13; every citation is verifiable.

## 1. Scope & motivation

<!-- drafted in Task 5 -->

## 2. Composites & XCT characterization — background

<!-- drafted in Task 5 -->

## 3. Standards & qualification

XCT is a primary nondestructive examination (NDE) modality for fiber-reinforced
polymer (FRP) composites in aerospace and defense, resolving porosity,
delaminations, fiber waviness, and inclusions that two-dimensional radiography
cannot [A1-10]. Its adoption is governed by interlocking technical standards,
accreditation, and personnel-qualification rules.

The foundational U.S. CT standards are maintained by ASTM Committee E07. ASTM
E1441 establishes imaging principles and a consistent vocabulary of CT
performance parameters [A1-1]; ASTM E1570 prescribes minimum procedural
requirements for fan-beam CT examination, written to be invoked on drawings or
contracts [A1-2]; and ASTM E1695 standardises measurement of spatial resolution
(via the Modulation Transfer Function) and contrast sensitivity, giving buyers
and suppliers quantitative tools for system qualification [A1-3]. For
composite-specific NDE, ASTM E2533 is the consensus guide covering CT among nine
methods for aerospace polymer-matrix composites, deliberately deferring
acceptance criteria to the Cognizant Engineering Organization [A1-4].

Internationally, the ISO 15708 series provides a harmonised framework:
ISO 15708-2 covers principles, equipment, and samples [A1-6], while ISO 15708-4
addresses qualification of CT system performance against task-specific
requirements [A1-5]. For dimensional CT, VDI/VDE 2630 Part 1.2 catalogues
influencing variables and uncertainty sources underpinning metrological
traceability [A1-8]. The Composite Materials Handbook CMH-17, Volume 3, is the
governing data-substantiation reference for aerospace polymer-matrix composites,
situating NDE within the broader qualification framework [A1-7].

Aerospace primes mandate NADCAP accreditation for NDT suppliers (audit criteria
AC7114), requiring inspection personnel certified to NAS 410 [A1-9]. A widely
applied acceptance heuristic for structural laminates is a porosity volume
fraction below roughly 2%, above which interlaminar shear strength degrades
measurably; XCT supplies the void morphology (size distribution, clustering)
that acoustic C-scan cannot. Nikishkov et al. established foundational XCT
methodology for three-dimensional void dimensioning in CFRP and quantified
systematic measurement errors [A1-13]; Elkolali et al. compared micro-CT at
3.5 µm with destructive acid digestion, confirming CT's advantage for void
morphology [A1-14]; and Galvez-Hernandez et al. showed that voxel sizes above
25 µm overestimate void content through partial-volume effects and that scans
shorter than ~2 min introduce noise-induced artefacts, establishing practical
scan-parameter bounds [A1-12]. Reviews by Naresh et al. and Gao et al. document
XCT's expanding role across process monitoring, fiber-orientation mapping, and
post-cure quality assurance [A1-10, A1-11].

## 4. Acquisition & preprocessing

X-ray computed tomography (XCT) provides three-dimensional maps of fiber
architecture, voids, and damage in fiber-reinforced polymer (FRP) composites
without destructive sectioning [A2-3]. Accurate phase segmentation is obstructed
by several acquisition-related artifacts that distort gray-level distributions:
beam hardening, ring artifacts, partial-volume effects, and photon-counting
noise [A2-1].

Beam hardening arises because polychromatic sources preferentially absorb
low-energy photons as the beam traverses the specimen, making reconstructed
interior attenuation appear systematically lower than at the surface (cupping);
in multi-constituent composites, differential hardening also produces bright
streaks at high-density phase boundaries. Corrections span physical
pre-filtration, projection-domain linearisation against material attenuation
models, and iterative post-reconstruction compensation [A2-1]. Ring artifacts
originate from detector gain non-uniformities; Münch et al. [A2-2] showed that a
combined wavelet–Fourier sinogram filter suppresses these stripe patterns while
preserving structure, a step now standard in FRP micro-CT workflows [A2-3].
Partial-volume effects emerge when the voxel dimension is comparable to the
fiber diameter (typically 5–15 µm for glass or carbon fibers), producing
intermediate boundary-voxel intensities that blur interfaces and bias volume
fraction estimates [A2-1, A2-6]. Photon-counting noise, compounded by the
intrinsically low contrast between carbon fibers and polymer matrix, further
degrades segmentation signal-to-noise ratio [A2-5].

Flat-field normalisation (dividing each projection by a dark-subtracted detector
reference) corrects gain inhomogeneity as a prerequisite; post-reconstruction
denoising by median or non-local-means filters — and, more recently,
unsupervised super-resolution enhancement [A2-9] — sharpens fiber boundaries
prior to segmentation.

## 5. Thresholding & segmentation

### Classical methods

Threshold-based segmentation remains the first-line approach in FRP tomography.
Otsu's method [A2-4] selects the global intensity threshold that minimises
intra-class variance and performs well when constituent phases produce separated
histogram modes, as in glass-fiber/epoxy systems; multi-Otsu generalisations
extend the criterion to three or more classes, delineating fiber tows, matrix,
and voids from a single histogram [A2-6]. Global thresholds fail when residual
beam-hardening gradients spatially shift intensity levels or when low
fiber–matrix contrast causes histogram overlap, common in carbon/epoxy
laminates [A2-5]. Locally adaptive methods address this by computing the
criterion within a sliding window; Bradley and Roth [A2-8] gave an O(N)
integral-image implementation, while Sauvola and Niblack formulations weight the
local standard deviation to suppress false positives. After binarisation,
touching fiber cross-sections are separated by a watershed transform seeded at
maxima of the Euclidean distance transform; Emerson et al. [A2-7] applied this
pipeline to individual fiber extraction and orientation in unidirectional glass
and carbon FRP at volume fractions up to 55%. Comparative analyses confirm that
greyscale thresholding with appropriate preprocessing suffices for
well-contrasted glass/polymer systems but degrades for carbon/epoxy composites
with irregular void morphology, motivating learning-based alternatives [A2-10].

### Deep learning and foundation models

Deep learning has substantially advanced volumetric segmentation of composites.
The encoder–decoder architecture of Ronneberger et al. [A3-1] established the
prevailing template, and its 3D extension by Çiçek et al. [A3-2] enables dense
voxel-wise prediction from sparse slice annotations — well-suited to the large,
partially labeled volumes typical of composite CT. The nnU-Net framework of
Isensee et al. [A3-3] automated topology, preprocessing, and hyperparameter
selection, becoming a practical default for industrial volumetric analysis.

Application to FRP composites poses distinctive challenges: extremely low
attenuation contrast between carbon fibers and polymer matrices, fiber diameters
approaching the voxel size, and labor-intensive volumetric annotation. Sinchuk
et al. [A3-4] applied U-Net to very-low-contrast carbon/epoxy woven μCT,
achieving ~4.7% Dice error from as few as 14 annotated slices; Badran et al.
[A3-5] segmented ceramic-matrix composites where fiber and matrix share
near-identical intensities, with a companion study [A3-6] standardising
evaluation against ground-truth references. de Siqueira et al. [A3-7] published
a reusable pipeline reaching Dice and Matthews coefficients near 98% with a
curated open dataset, and Konopczyński et al. [A3-8] addressed instance-level
fiber separation via 3D metric embedding. Recent work adds attention-based and
hybrid architectures: Guo et al. [A3-9] paired unsupervised CycleGAN enhancement
with an instance-segmentation head for fast time-resolved synchrotron CT, and
Chen et al. [A3-10] showed a Swin-Transformer encoder outperforming CNN
baselines for multi-phase XCT segmentation.

Foundation and promptable models are an emerging paradigm. The Segment Anything
Model [A3-11] enables prompt-driven zero-shot segmentation, and its volumetric
successor SAM-Med3D [A3-12] extends this to 3D CT volumes. Tabassum and Ziabari
[A3-13] showed that parameter-efficient fine-tuning of SAM on GAN-generated
synthetic industrial CT closes much of the domain gap for material inspection.
For label-free composite training, Friemann et al. [A3-14] generated
physics-rendered synthetic CT of 3D-textile CFRP with automatic masks, and
Yunker et al. [A3-15] initialised segmentation from voxel-clustering pseudo-labels
that self-correct iteratively, improving mean IoU without manual annotation.

## 6. Fiber orientation & morphometry

<!-- drafted in Task 3 -->

## 7. Fibre tracking & network analysis

<!-- drafted in Task 3 -->

## 8. Deformation measurement — DVC & DIC

<!-- drafted in Task 3 -->

## 9. Prediction of mechanical behavior

<!-- drafted in Task 3 -->

## 10. Digital twins for composites

<!-- drafted in Task 4 -->

## 11. Cross-cutting concerns

<!-- drafted in Task 4 -->

## 12. Synthesis & gap analysis

<!-- drafted in Task 5 -->

## 13. References

<!-- assembled in Task 6 -->
