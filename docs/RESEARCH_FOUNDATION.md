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

Fiber orientation is the microstructural variable most tightly coupled to the
stiffness and strength anisotropy of a composite, and its quantitative
description in continuum models rests on the orientation-tensor formalism
introduced by Advani and Tucker, who represented the orientation distribution
function by its even-order moment tensors — most practically the second-order
orientation tensor — enabling direct coupling between measured microstructure
and process or structural simulation [A4-1]. Before volumetric CT was routine,
three-dimensional orientation was inferred stereologically from the elliptical
cross-sections fibers leave on a polished 2D section; Bay and Tucker derived
orientation-dependent weighting functions and confidence limits to correct the
sampling bias inherent in such area-based estimates [A4-2], a correction logic
that still underlies section-based validation of CT-derived results.

Volumetric CT enabled voxel-wise orientation estimation without prior fiber
segmentation via the gradient structure tensor, whose local eigenvector
decomposition yields the dominant fiber direction and an eigenvalue-based
anisotropy measure at each voxel. Krause et al. combined the structure tensor
with a local X-ray transform for simultaneous denoising and orientation
mapping in ceramic- and glass-fiber composites [A4-3], building on practical
algorithms for fiber orientation estimation from 3D image data established by
Robb, Wirjadi, and Schladitz [A4-4]. The approach scales to industrial
components: Baranowski et al. combined region-of-interest CT with
texture-orientation analysis to map local fiber orientation across a large
automotive bearing part, validating consistency across scan resolutions
against simulated orientation tensors [A4-5], while Karamov et al. benchmarked
structure-tensor orientation against high-fidelity, fully segmented
fiber-identification methods in random-fiber composites, comparing the two
families of method [A4-6]. Where individual fibers are already segmented,
principal component analysis of each fiber's voxel point cloud offers an
alternative, geometry-driven estimator: Salling et al. used PCA-based
inclination segmentation to recover per-fiber orientation directly from CT
[A4-7].

Morphometric quantification — diameter, length, and tortuosity — complements
orientation as direct input to micromechanical models. Zanini and Carmignato
developed a physical reference object to establish measurement traceability
for CT-based fiber length determination, addressing a persistent metrological
gap [A4-8]. For curvilinear descriptors, Gomarasca et al. proposed a
three-tier hierarchy of microstructural descriptors — single-fiber tortuosity,
fiber-group behavior, and fiber-network interconnectivity — validated on CT of
unidirectional carbon/PEEK tape [A4-9], and Julià i Juanola et al. addressed
the ill-posedness of direct curvature/waviness computation on noisy,
irregularly sampled centerlines with a Frenet–Serret-based algorithm that uses
frequency-limited Gaussian low-pass filtering to separate genuine waviness
from sampling noise [A4-10].

## 7. Fibre tracking & network analysis

Reconstructing individual fiber centerlines from a segmented CT volume is a
prerequisite for per-fiber statistics, finite-element mesh generation, and
network-level connectivity analysis. Two complementary strategies dominate:
tracking, which grows a trajectory fiber-by-fiber through the volume, and
skeletonization, which thins the entire binarized fiber phase to a
one-voxel-wide medial-axis representation in a single pass. Czabaj, Riccio,
and Whitacre demonstrated the tracking approach, combining 2D template
matching for fiber-center detection with a multi-target Kalman-filter
tracking estimator to reconstruct individual fiber paths through a
sub-micron-resolution graphite/epoxy volume for direct numerical mesh
reconstruction [A4-11]. Huang et al. demonstrated the skeletonization
alternative, extracting curved fiber centerlines from micro-CT by linking
neighboring skeleton segments according to orientation and radius similarity,
a strategy validated on a highly porous sintered metal fiber sheet [A4-12].

Skeletonization alone under-resolves regions where fibers physically touch,
since topology-preserving thinning cannot separate contacting branches;
Depriester et al. addressed this with a general fiber-separation algorithm,
applicable across fiber geometries (straight or woven, circular or
non-circular cross-section), that localizes fiber–fiber contacts independent
of fiber orientation, extending centerline-based tracking to densely packed
and textile composites [A4-13]. Once individual centerlines are recovered,
they can be assembled into a graph in which fibers, contacts, and crossings
become nodes and edges; Gomarasca et al.'s fiber-network-interconnectivity
descriptor is an instance of this representation, treating the assembly's
connectivity — rather than any single fiber's path — as the top tier of
microstructural complexity [A4-9]. Fiber-tracking and network-graph outputs
feed directly into as-built finite-element models, closing the loop between
CT-observed microstructure and mechanical simulation.

## 8. Deformation measurement — DVC & DIC

Digital image correlation (DIC) and its volumetric extension, digital volume
correlation (DVC), register grey-level patterns between a reference and a
deformed image (or tomogram) to recover full-field displacement and strain
without contact instrumentation. The subset-based formulation traces to
Sutton et al. [A5-1], who established gradient-based subpixel registration of
speckled 2D surfaces; Bay et al. [A5-2] extended the correlation principle to
volumetric X-ray CT data, using the naturally occurring trabecular texture of
bone as an internal speckle pattern and establishing DVC as a distinct
discipline from surface DIC.

Correlation algorithms divide into local and global formulations. Local
(subset-based) methods independently register a regular grid of subvolumes or
subsets against the reference image, are embarrassingly parallel, but yield
noisy, discontinuous displacement fields with no inter-subset continuity
constraint. Global methods instead solve a single minimisation over a
finite-element mesh spanning the whole domain, enforcing displacement
continuity (or, with enriched bases, controlled discontinuity) across element
boundaries; Hild and Roux [A5-3] showed global correlation generally achieves
better spatial-resolution/noise trade-offs than local correlation under
matched conditions. Buljac et al. [A5-4] unify both paradigms within a single
DVC framework and catalogue bias and uncertainty sources — spatial
resolution, mesh/subset size, interpolation order, and image noise — that
propagate into derived strain fields.

Two further axes distinguish implementations. FFT-based approaches use
Fourier cross-correlation (or phase correlation) to obtain a rapid, robust
initial integer-voxel displacement estimate ahead of iterative subvoxel
refinement (e.g., inverse-compositional Gauss-Newton); Bar-Kochba et al.'s
fast iterative DVC (FIDVC) [A5-5] combined this with multi-pass refinement to
reach large-deformation convergence in one to two minutes on commodity
hardware. Optical-flow-based approaches instead treat correlation as dense
per-voxel motion estimation; Wong et al.'s VolRAFT [A5-6] adapts the 2D RAFT
optical-flow network to 3D, learning a displacement field directly from
paired synchrotron micro-CT volumes and matching iterative DVC accuracy at
inference-time cost, though — trained on bone-implant data — its transfer to
fiber-composite microstructure is untested.

Open-source tooling has substantially lowered the barrier to DVC/DIC adoption
outside commercial packages such as LaVision DaVis, Correlated Solutions
VIC-3D, and GOM Correlate. spam [A5-7] is a Python library offering local,
global, and discrete (particle-level) correlation plus multimodal
registration, widely used for granular and porous media and increasingly for
composites; muDIC [A5-8] is a pure-Python 2D DIC toolkit using B-spline
finite-element discretisation with an integrated synthetic-image generator
for virtual experiments; Ncorr [A5-9] remains the most widely cited free
subset-based 2D DIC package, offering a MATLAB/C++ alternative to commercial
2D systems.

Validation practice centres on interlaboratory benchmarking. The DVC
Challenge circulated common CT datasets across multiple laboratories,
algorithms, and scanners to isolate displacement/strain measurement
uncertainty from true material response; Croom et al. [A5-10] report an
update quantifying how CT equipment and scan parameters alone modulate DVC
error across participating labs, underscoring that DVC-derived strain fields
require reported, scan-specific uncertainty bounds before use as
model-validation ground truth.

For FRP composites specifically, surface DIC is routine for coupon-level
strain mapping (off-axis tension, open-hole, notched specimens), while DVC
adoption is more recent and growing: Mehdikhani et al. [A5-11] applied DVC to
meso/micro in-situ tensile damage in CFRP directly from tomograms, exploiting
the fiber architecture itself as a natural speckle; Wang et al. [A5-12]
coupled DVC with deep-learning damage characterisation for in-situ CT
testing. Holmes et al.'s review [A5-13] surveys DIC/DVC applications
specifically across fiber-reinforced composites, and Jiang et al. [A5-14]
situate DVC within a broader CT-deep-learning-finite-element pipeline for
composite defect analysis, reflecting a broader convergence toward integrated
analysis-to-simulation workflows.

## 9. Prediction of mechanical behavior

Predicting mechanical behavior from CT-derived microstructure connects
segmentation and fiber-architecture measurement to structural performance,
closing the loop between as-built geometry and design allowables. The
theoretical basis is classical micromechanics: Eshelby's solution for the
elastic field of an ellipsoidal inclusion in an infinite matrix [A6-1]
underlies mean-field homogenization schemes, most notably the Mori–Tanaka
method, which estimates the average stress in the matrix phase to predict
effective stiffness of two-phase composites at arbitrary volume fraction
[A6-2]. The semi-empirical Halpin–Tsai equations, derived from Hill's
self-consistent framework, offer a computationally trivial alternative widely
used for quick property estimation from fiber aspect ratio and volume
fraction [A6-3]. Kanouté et al. review how these closed-form mean-field
methods relate to full-field computational homogenization —
asymptotic/periodic homogenization and unit-cell finite-element (FE) analysis
— establishing the accuracy-versus-cost spectrum that CT-informed models must
navigate [A6-4].

Coupling homogenization with structural FE analysis is most rigorously
realized in concurrent multiscale schemes: the FE² method of Feyel and
Chaboche nests a microscale unit-cell FE solve at every macroscale
integration point, propagating heterogeneous constitutive response without a
closed-form homogenized law [A6-5]. Llorca et al. articulate a bottom-up
"virtual testing" roadmap in which constituent- and microscale properties,
measured or imaged directly, feed a hierarchy of models toward
structural-scale prediction, reducing reliance on coupon testing [A6-6]. CT
imaging increasingly supplies the geometric input to this hierarchy directly:
Sinchuk et al. built multi-layer unit-cell FE models from segmented X-ray CT
of carbon-fiber textile composites, meshing the as-scanned tow architecture
to homogenize elastic properties and compare against idealized geometry
[A6-7]. In ceramic-matrix composites, Ai et al. combined image-based FE
models constructed from CT with in-situ XCT damage observation to track
stress redistribution and matrix cracking through loading history in woven
C/SiC, validating failure prediction directly against imaged crack paths
[A6-8].

Machine-learning surrogates are displacing repeated FE solves where many
microstructure realizations or load cases must be evaluated. Convolutional
networks trained on segmented microstructure images predict homogenized
elastic and strength properties orders of magnitude faster than direct FE, as
demonstrated by Sun et al. for fiber-reinforced polymer cross-sections
[A6-9], while graph neural networks operating directly on fiber-topology
graphs extend this to stiffness and fracture-initiation prediction under high
phase-contrast, per Caliskan et al. [A6-10]. Huang et al. survey the broader
landscape, organizing ML applications in composite micromechanics into
microstructure–property mapping, multiphysics field prediction,
constitutive-model learning, and inverse microstructure design [A6-11].
Fatigue-life prediction, historically reliant on empirical S–N curves, has
adopted similar data-driven regressors: Hemanth Kumar and Swamy trained
artificial neural networks on experimental fatigue data for glass-fiber/epoxy
laminates, reporting improved life prediction over classical curve-fitting
[A6-12].

Because both mean-field homogenization and learned surrogates are
approximations calibrated on finite data, uncertainty quantification (UQ) is
increasingly treated as integral to prediction rather than a post-hoc
addendum. Balokas et al. applied Bayesian inverse UQ to calibrate microscale
constitutive parameters against measured transverse tensile response,
propagating experimental scatter into predicted composite strength
distributions [A6-13]; García-Merino et al. used polynomial chaos expansion
as a surrogate for periodic unit-cell homogenization, enabling efficient
forward propagation of constituent-property uncertainty into effective
stiffness statistics without repeated FE solves [A6-14].

## 10. Digital twins for composites

<!-- drafted in Task 4 -->

## 11. Cross-cutting concerns

<!-- drafted in Task 4 -->

## 12. Synthesis & gap analysis

<!-- drafted in Task 5 -->

## 13. References

<!-- assembled in Task 6 -->
