# Research Foundation

A literature survey establishing the academic and research background for Fiber
Tracer, spanning the analysis → prediction → digital-twin continuum for X-ray
computed tomography (XCT) of fiber-reinforced composites, framed for
high-reliability (aerospace/defense) use. References are numbered and listed in
section 13; every citation is verifiable.

## 1. Scope & motivation

This survey establishes the research foundation for a software pipeline that
takes X-ray computed tomography (XCT) of fiber-reinforced polymer (FRP)
composites through segmentation, fiber-level measurement, deformation
quantification, and mechanical-property prediction, framed for
high-reliability aerospace/defense use where NDE evidence must trace to
governing standards and qualification records [4, 7]. It covers the
continuum from acquisition and segmentation (§3–§5), through fiber
orientation, morphometry, and network-level tracking (§6–§7), to full-field
deformation measurement (§8), mechanical-behavior prediction (§9), and the
digital-twin and cross-cutting concerns that connect measured microstructure
back to a qualification decision (§10–§11). The scope is deliberately bounded
to methods with a demonstrated or plausible path to CT-derived, per-part
analysis of FRP composites; adjacent fields (medical CT, general materials
informatics) are cited only where they establish a technique or an
evidenced risk (e.g., domain shift [82]) that transfers directly.

## 2. Composites & XCT characterization — background

Fiber-reinforced polymer composites achieve high specific stiffness and
strength by embedding continuous or discontinuous reinforcing fibers
(typically carbon or glass) in a polymer matrix, but their performance is
governed as much by processing-induced microstructural disorder — fiber
misalignment and waviness, void content, tow nesting and crimp in woven
architectures — as by the nominal constituent properties [38, 46]. Unlike
metals, FRP composites cannot be qualified by coupon testing alone at scale,
because damage initiation and failure are locally controlled by this
disorder; NDE must therefore resolve three-dimensional microstructure, not
merely detect gross defects. XCT is the modality of choice because it is the
only common technique that non-destructively resolves internal fiber
architecture, porosity, and delamination in three dimensions at the
fiber-diameter scale [10]. Realizing that potential in practice, however,
depends on the entire chain surveyed here: correcting acquisition artifacts
that would otherwise bias measurement (§4), converting reconstructed
grayscale volumes into a labeled fiber/matrix/void representation (§5),
extracting orientation and per-fiber geometry from that representation
(§6–§7), and — where the goal is validating or predicting mechanical
response rather than only characterizing geometry — coupling CT-derived
microstructure to deformation measurement (§8) and predictive models (§9).

## 3. Standards & qualification

XCT is a primary nondestructive examination (NDE) modality for fiber-reinforced
polymer (FRP) composites in aerospace and defense, resolving porosity,
delaminations, fiber waviness, and inclusions that two-dimensional radiography
cannot [10]. Its adoption is governed by interlocking technical standards,
accreditation, and personnel-qualification rules.

The foundational U.S. CT standards are maintained by ASTM Committee E07. ASTM
E1441 establishes imaging principles and a consistent vocabulary of CT
performance parameters [1]; ASTM E1570 prescribes minimum procedural
requirements for fan-beam CT examination, written to be invoked on drawings or
contracts [2]; and ASTM E1695 standardises measurement of spatial resolution
(via the Modulation Transfer Function) and contrast sensitivity, giving buyers
and suppliers quantitative tools for system qualification [3]. For
composite-specific NDE, ASTM E2533 is the consensus guide covering CT among nine
methods for aerospace polymer-matrix composites, deliberately deferring
acceptance criteria to the Cognizant Engineering Organization [4].

Internationally, the ISO 15708 series provides a harmonised framework:
ISO 15708-2 covers principles, equipment, and samples [6], while ISO 15708-4
addresses qualification of CT system performance against task-specific
requirements [5]. For dimensional CT, VDI/VDE 2630 Part 1.2 catalogues
influencing variables and uncertainty sources underpinning metrological
traceability [8]. The Composite Materials Handbook CMH-17, Volume 3, is the
governing data-substantiation reference for aerospace polymer-matrix composites,
situating NDE within the broader qualification framework [7].

Aerospace primes mandate NADCAP accreditation for NDT suppliers (audit criteria
AC7114), requiring inspection personnel certified to NAS 410 [9]. A widely
applied acceptance heuristic for structural laminates is a porosity volume
fraction below roughly 2%, above which interlaminar shear strength degrades
measurably; XCT supplies the void morphology (size distribution, clustering)
that acoustic C-scan cannot. Nikishkov et al. established foundational XCT
methodology for three-dimensional void dimensioning in CFRP and quantified
systematic measurement errors [13]; Elkolali et al. compared micro-CT at
3.5 µm with destructive acid digestion, confirming CT's advantage for void
morphology [14]; and Galvez-Hernandez et al. showed that voxel sizes above
25 µm overestimate void content through partial-volume effects and that scans
shorter than ~2 min introduce noise-induced artefacts, establishing practical
scan-parameter bounds [12]. Reviews by Naresh et al. and Gao et al. document
XCT's expanding role across process monitoring, fiber-orientation mapping, and
post-cure quality assurance [10, 11].

## 4. Acquisition & preprocessing

X-ray computed tomography (XCT) provides three-dimensional maps of fiber
architecture, voids, and damage in fiber-reinforced polymer (FRP) composites
without destructive sectioning [15]. Accurate phase segmentation is obstructed
by several acquisition-related artifacts that distort gray-level distributions:
beam hardening, ring artifacts, partial-volume effects, and photon-counting
noise [15].

Beam hardening arises because polychromatic sources preferentially absorb
low-energy photons as the beam traverses the specimen, making reconstructed
interior attenuation appear systematically lower than at the surface (cupping);
in multi-constituent composites, differential hardening also produces bright
streaks at high-density phase boundaries. Corrections span physical
pre-filtration, projection-domain linearisation against material attenuation
models, and iterative post-reconstruction compensation [15]. Ring artifacts
originate from detector gain non-uniformities; Münch et al. [16] showed that a
combined wavelet–Fourier sinogram filter suppresses these stripe patterns while
preserving structure, a step now standard in FRP micro-CT workflows [15].
Partial-volume effects emerge when the voxel dimension is comparable to the
fiber diameter (typically 5–15 µm for glass or carbon fibers), producing
intermediate boundary-voxel intensities that blur interfaces and bias volume
fraction estimates [15, 18]. Photon-counting noise, compounded by the
intrinsically low contrast between carbon fibers and polymer matrix, further
degrades segmentation signal-to-noise ratio [15].

Flat-field normalisation (dividing each projection by a dark-subtracted detector
reference) corrects gain inhomogeneity as a prerequisite; post-reconstruction
denoising by median or non-local-means filters — and, more recently,
unsupervised super-resolution enhancement [21] — sharpens fiber boundaries
prior to segmentation.

## 5. Thresholding & segmentation

### Classical methods

Threshold-based segmentation remains the first-line approach in FRP tomography.
Otsu's method [17] selects the global intensity threshold that minimises
intra-class variance and performs well when constituent phases produce separated
histogram modes, as in glass-fiber/epoxy systems; multi-Otsu generalisations
extend the criterion to three or more classes, delineating fiber tows, matrix,
and voids from a single histogram [18]. Global thresholds fail when residual
beam-hardening gradients spatially shift intensity levels or when low
fiber–matrix contrast causes histogram overlap, common in carbon/epoxy
laminates [15]. Locally adaptive methods address this by computing the
criterion within a sliding window; Bradley and Roth [20] gave an O(N)
integral-image implementation, while Sauvola and Niblack formulations weight the
local standard deviation to suppress false positives. After binarisation,
touching fiber cross-sections are separated by a watershed transform seeded at
maxima of the Euclidean distance transform; Emerson et al. [19] applied this
pipeline to individual fiber extraction and orientation in unidirectional glass
and carbon FRP at volume fractions up to 55%. Comparative analyses confirm that
greyscale thresholding with appropriate preprocessing suffices for
well-contrasted glass/polymer systems but degrades for carbon/epoxy composites
with irregular void morphology, motivating learning-based alternatives [22].

### Deep learning and foundation models

Deep learning has substantially advanced volumetric segmentation of composites.
The encoder–decoder architecture of Ronneberger et al. [23] established the
prevailing template, and its 3D extension by Çiçek et al. [24] enables dense
voxel-wise prediction from sparse slice annotations — well-suited to the large,
partially labeled volumes typical of composite CT. The nnU-Net framework of
Isensee et al. [25] automated topology, preprocessing, and hyperparameter
selection, becoming a practical default for industrial volumetric analysis.

Application to FRP composites poses distinctive challenges: extremely low
attenuation contrast between carbon fibers and polymer matrices, fiber diameters
approaching the voxel size, and labor-intensive volumetric annotation. Sinchuk
et al. [26] applied U-Net to very-low-contrast carbon/epoxy woven μCT,
achieving ~4.7% Dice error from as few as 14 annotated slices; Badran et al.
[27] segmented ceramic-matrix composites where fiber and matrix share
near-identical intensities, with a companion study [28] standardising
evaluation against ground-truth references. de Siqueira et al. [29] published
a reusable pipeline reaching Dice and Matthews coefficients near 98% with a
curated open dataset, and Konopczyński et al. [30] addressed instance-level
fiber separation via 3D metric embedding. Recent work adds attention-based and
hybrid architectures: Guo et al. [31] paired unsupervised CycleGAN enhancement
with an instance-segmentation head for fast time-resolved synchrotron CT, and
Chen et al. [32] showed a Swin-Transformer encoder outperforming CNN
baselines for multi-phase XCT segmentation.

Foundation and promptable models are an emerging paradigm. The Segment Anything
Model [33] enables prompt-driven zero-shot segmentation, and its volumetric
successor SAM-Med3D [34] extends this to 3D CT volumes. Tabassum and Ziabari
[35] showed that parameter-efficient fine-tuning of SAM on GAN-generated
synthetic industrial CT closes much of the domain gap for material inspection.
For label-free composite training, Friemann et al. [36] generated
physics-rendered synthetic CT of 3D-textile CFRP with automatic masks, and
Yunker et al. [37] initialised segmentation from voxel-clustering pseudo-labels
that self-correct iteratively, improving mean IoU without manual annotation.

## 6. Fiber orientation & morphometry

Fiber orientation is the microstructural variable most tightly coupled to the
stiffness and strength anisotropy of a composite, and its quantitative
description in continuum models rests on the orientation-tensor formalism
introduced by Advani and Tucker, who represented the orientation distribution
function by its even-order moment tensors — most practically the second-order
orientation tensor — enabling direct coupling between measured microstructure
and process or structural simulation [38]. Before volumetric CT was routine,
three-dimensional orientation was inferred stereologically from the elliptical
cross-sections fibers leave on a polished 2D section; Bay and Tucker derived
orientation-dependent weighting functions and confidence limits to correct the
sampling bias inherent in such area-based estimates [39], a correction logic
that still underlies section-based validation of CT-derived results.

Volumetric CT enabled voxel-wise orientation estimation without prior fiber
segmentation via the gradient structure tensor, whose local eigenvector
decomposition yields the dominant fiber direction and an eigenvalue-based
anisotropy measure at each voxel. Krause et al. combined the structure tensor
with a local X-ray transform for simultaneous denoising and orientation
mapping in ceramic- and glass-fiber composites [40], building on practical
algorithms for fiber orientation estimation from 3D image data established by
Robb, Wirjadi, and Schladitz [41]. The approach scales to industrial
components: Baranowski et al. combined region-of-interest CT with
texture-orientation analysis to map local fiber orientation across a large
automotive bearing part, validating consistency across scan resolutions
against simulated orientation tensors [42], while Karamov et al. benchmarked
structure-tensor orientation against high-fidelity, fully segmented
fiber-identification methods in random-fiber composites, comparing the two
families of method [43]. Where individual fibers are already segmented,
principal component analysis of each fiber's voxel point cloud offers an
alternative, geometry-driven estimator: Salling et al. used PCA-based
inclination segmentation to recover per-fiber orientation directly from CT
[44].

Morphometric quantification — diameter, length, and tortuosity — complements
orientation as direct input to micromechanical models. Zanini and Carmignato
developed a physical reference object to establish measurement traceability
for CT-based fiber length determination, addressing a persistent metrological
gap [45]. For curvilinear descriptors, Gomarasca et al. proposed a
three-tier hierarchy of microstructural descriptors — single-fiber tortuosity,
fiber-group behavior, and fiber-network interconnectivity — validated on CT of
unidirectional carbon/PEEK tape [46], and Julià i Juanola et al. addressed
the ill-posedness of direct curvature/waviness computation on noisy,
irregularly sampled centerlines with a Frenet–Serret-based algorithm that uses
frequency-limited Gaussian low-pass filtering to separate genuine waviness
from sampling noise [47].

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
reconstruction [48]. Huang et al. demonstrated the skeletonization
alternative, extracting curved fiber centerlines from micro-CT by linking
neighboring skeleton segments according to orientation and radius similarity,
a strategy validated on a highly porous sintered metal fiber sheet [49].

Skeletonization alone under-resolves regions where fibers physically touch,
since topology-preserving thinning cannot separate contacting branches;
Depriester et al. addressed this with a general fiber-separation algorithm,
applicable across fiber geometries (straight or woven, circular or
non-circular cross-section), that localizes fiber–fiber contacts independent
of fiber orientation, extending centerline-based tracking to densely packed
and textile composites [50]. Once individual centerlines are recovered,
they can be assembled into a graph in which fibers, contacts, and crossings
become nodes and edges; Gomarasca et al.'s fiber-network-interconnectivity
descriptor is an instance of this representation, treating the assembly's
connectivity — rather than any single fiber's path — as the top tier of
microstructural complexity [46]. Fiber-tracking and network-graph outputs
feed directly into as-built finite-element models, closing the loop between
CT-observed microstructure and mechanical simulation.

## 8. Deformation measurement — DVC & DIC

Digital image correlation (DIC) and its volumetric extension, digital volume
correlation (DVC), register grey-level patterns between a reference and a
deformed image (or tomogram) to recover full-field displacement and strain
without contact instrumentation. The subset-based formulation traces to
Sutton et al. [51], who established gradient-based subpixel registration of
speckled 2D surfaces; Bay et al. [52] extended the correlation principle to
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
boundaries; Hild and Roux [53] showed global correlation generally achieves
better spatial-resolution/noise trade-offs than local correlation under
matched conditions. Buljac et al. [54] unify both paradigms within a single
DVC framework and catalogue bias and uncertainty sources — spatial
resolution, mesh/subset size, interpolation order, and image noise — that
propagate into derived strain fields.

Two further axes distinguish implementations. FFT-based approaches use
Fourier cross-correlation (or phase correlation) to obtain a rapid, robust
initial integer-voxel displacement estimate ahead of iterative subvoxel
refinement (e.g., inverse-compositional Gauss-Newton); Bar-Kochba et al.'s
fast iterative DVC (FIDVC) [55] combined this with multi-pass refinement to
reach large-deformation convergence in one to two minutes on commodity
hardware. Optical-flow-based approaches instead treat correlation as dense
per-voxel motion estimation; Wong et al.'s VolRAFT [56] adapts the 2D RAFT
optical-flow network to 3D, learning a displacement field directly from
paired synchrotron micro-CT volumes and matching iterative DVC accuracy at
inference-time cost, though — trained on bone-implant data — its transfer to
fiber-composite microstructure is untested.

Open-source tooling has substantially lowered the barrier to DVC/DIC adoption
outside commercial packages such as LaVision DaVis, Correlated Solutions
VIC-3D, and GOM Correlate. spam [57] is a Python library offering local,
global, and discrete (particle-level) correlation plus multimodal
registration, widely used for granular and porous media and increasingly for
composites; muDIC [58] is a pure-Python 2D DIC toolkit using B-spline
finite-element discretisation with an integrated synthetic-image generator
for virtual experiments; Ncorr [59] remains the most widely cited free
subset-based 2D DIC package, offering a MATLAB/C++ alternative to commercial
2D systems.

Validation practice centres on interlaboratory benchmarking. The DVC
Challenge circulated common CT datasets across multiple laboratories,
algorithms, and scanners to isolate displacement/strain measurement
uncertainty from true material response; Croom et al. [60] report an
update quantifying how CT equipment and scan parameters alone modulate DVC
error across participating labs, underscoring that DVC-derived strain fields
require reported, scan-specific uncertainty bounds before use as
model-validation ground truth.

For FRP composites specifically, surface DIC is routine for coupon-level
strain mapping (off-axis tension, open-hole, notched specimens), while DVC
adoption is more recent and growing: Mehdikhani et al. [61] applied DVC to
meso/micro in-situ tensile damage in CFRP directly from tomograms, exploiting
the fiber architecture itself as a natural speckle; Wang et al. [62]
coupled DVC with deep-learning damage characterisation for in-situ CT
testing. Holmes et al.'s review [63] surveys DIC/DVC applications
specifically across fiber-reinforced composites, and Jiang et al. [64]
situate DVC within a broader CT-deep-learning-finite-element pipeline for
composite defect analysis, reflecting a broader convergence toward integrated
analysis-to-simulation workflows.

## 9. Prediction of mechanical behavior

Predicting mechanical behavior from CT-derived microstructure connects
segmentation and fiber-architecture measurement to structural performance,
closing the loop between as-built geometry and design allowables. The
theoretical basis is classical micromechanics: Eshelby's solution for the
elastic field of an ellipsoidal inclusion in an infinite matrix [65]
underlies mean-field homogenization schemes, most notably the Mori–Tanaka
method, which estimates the average stress in the matrix phase to predict
effective stiffness of two-phase composites at arbitrary volume fraction
[66]. The semi-empirical Halpin–Tsai equations, derived from Hill's
self-consistent framework, offer a computationally trivial alternative widely
used for quick property estimation from fiber aspect ratio and volume
fraction [67]. Kanouté et al. review how these closed-form mean-field
methods relate to full-field computational homogenization —
asymptotic/periodic homogenization and unit-cell finite-element (FE) analysis
— establishing the accuracy-versus-cost spectrum that CT-informed models must
navigate [68].

Coupling homogenization with structural FE analysis is most rigorously
realized in concurrent multiscale schemes: the FE² method of Feyel and
Chaboche nests a microscale unit-cell FE solve at every macroscale
integration point, propagating heterogeneous constitutive response without a
closed-form homogenized law [69]. Llorca et al. articulate a bottom-up
"virtual testing" roadmap in which constituent- and microscale properties,
measured or imaged directly, feed a hierarchy of models toward
structural-scale prediction, reducing reliance on coupon testing [70]. CT
imaging increasingly supplies the geometric input to this hierarchy directly:
Sinchuk et al. built multi-layer unit-cell FE models from segmented X-ray CT
of carbon-fiber textile composites, meshing the as-scanned tow architecture
to homogenize elastic properties and compare against idealized geometry
[71]. In ceramic-matrix composites, Ai et al. combined image-based FE
models constructed from CT with in-situ XCT damage observation to track
stress redistribution and matrix cracking through loading history in woven
C/SiC, validating failure prediction directly against imaged crack paths
[72].

Machine-learning surrogates are displacing repeated FE solves where many
microstructure realizations or load cases must be evaluated. Convolutional
networks trained on segmented microstructure images predict homogenized
elastic and strength properties orders of magnitude faster than direct FE, as
demonstrated by Sun et al. for fiber-reinforced polymer cross-sections
[73], while graph neural networks operating directly on fiber-topology
graphs extend this to stiffness and fracture-initiation prediction under high
phase-contrast, per Caliskan et al. [74]. Huang et al. survey the broader
landscape, organizing ML applications in composite micromechanics into
microstructure–property mapping, multiphysics field prediction,
constitutive-model learning, and inverse microstructure design [75].
Fatigue-life prediction, historically reliant on empirical S–N curves, has
adopted similar data-driven regressors: Hemanth Kumar and Swamy trained
artificial neural networks on experimental fatigue data for glass-fiber/epoxy
laminates, reporting improved life prediction over classical curve-fitting
[76].

Because both mean-field homogenization and learned surrogates are
approximations calibrated on finite data, uncertainty quantification (UQ) is
increasingly treated as integral to prediction rather than a post-hoc
addendum. Balokas et al. applied Bayesian inverse UQ to calibrate microscale
constitutive parameters against measured transverse tensile response,
propagating experimental scatter into predicted composite strength
distributions [77]; García-Merino et al. used polynomial chaos expansion
as a surrogate for periodic unit-cell homogenization, enabling efficient
forward propagation of constituent-property uncertainty into effective
stiffness statistics without repeated FE solves [78].

## 10. Digital twins for composites

The digital twin concept — a virtual representation of an individual physical
asset, kept synchronized with it through sensor and inspection data over the
asset's lifecycle — originated in aerospace structural life prediction. Tuegel
et al. proposed reengineering aircraft structural life prediction around an
"ultrahigh fidelity model of individual aircraft by tail number," integrating
flight-condition-driven stress and thermal computation with damage and
material-state evolution to assure structural integrity on a per-airframe
basis [79]. For fiber-reinforced composites specifically, the digital twin
premise depends on faithfully representing the as-manufactured microstructure
rather than the nominal as-designed geometry, because mechanical response is
disproportionately sensitive to processing-induced disorder — fiber
misalignment, waviness, void content, and tow nesting — that CT is uniquely
positioned to capture. Recent work targets this microscale coupling directly:
Hearley et al. describe a toolset combining convolutional segmentation with
prompt-based annotation to accelerate the extraction of microscopy-derived
microstructure into multiscale material digital twins for fiber-reinforced
composites, aiming to lower the annotation burden that otherwise limits
per-part digital twin construction [80]. More broadly in manufacturing,
systematic reviews of digital twins in additive manufacturing catalog the
same underlying pattern — physical-to-virtual data flow, model updating from
in-process sensing, and use of the twin for process control and quality
prediction — while flagging scalability, data-quality integration, and
real-time computational cost as persistent barriers [81], concerns that
transfer directly to CT-based composite twins given the size of volumetric CT
datasets and the cost of high-fidelity finite-element or DVC-based updating.
As-manufactured-to-as-designed reconciliation, the process of mapping
measured dimensions, geometries, and internal features from registered 3D
imaging data onto the nominal design model, is the mechanism by which a
digital twin departs from a purely theoretical model and instead reflects the
specific part under evaluation; this reconciliation step, together with data
assimilation and Bayesian model updating to fuse sparse sensor or inspection
observations with physics-based predictions, remains an active area with
limited maturity for composite structures specifically, most published
demonstrations still being at coupon or subcomponent scale rather than full
airframe scale.

## 11. Cross-cutting concerns

Three concerns cut across every stage of the CT-to-simulation pipeline
surveyed above: domain generalization of learned models, validation and
benchmarking practice, and reproducibility/traceability of the analysis
chain. Domain shift — degraded model performance when a segmentation or
reconstruction network trained on one scanner, material, or acquisition
protocol is applied to another — is a known risk wherever CT feature
extraction relies on learned models, and industrial XCT is no exception: a
review of machine learning in industrial X-ray CT surveys deep-learning use
throughout reconstruction, segmentation, and feature characterization and
notes that most reported models are trained and validated on narrow,
instrument-specific datasets, leaving open how well they transfer across
scanner vendors, voxel sizes, or fiber architectures [82]. Validation and
benchmarking practice in CT more generally is hampered by a scarcity of
openly available, task-complete datasets and by inconsistent or undocumented
implementations of reconstruction and evaluation methods; Polevoy et al.
argue explicitly that the field's benchmarking culture undermines
reproducibility because published comparisons rarely share raw projection
data, ground-truth reconstructions, or full method implementations, making
independent replication difficult even when results are reported honestly
[83]. For a pipeline that runs from raw CT projections through
segmentation, fiber-orientation and DVC measurement, to structural
prediction, this compounds: each stage's validation depends on artifacts
(phantom scans, ground-truth microstructures, reference mechanical tests)
that are seldom published alongside the methods that use them, and
traceability from a qualification decision back to the specific scan,
reconstruction parameters, and model version used to support it is rarely
demonstrated end to end in the composites literature, as distinct from being
asserted as a goal.

## 12. Synthesis & gap analysis

The literature surveyed above describes a continuum — acquisition and
segmentation feed fiber-level measurement, which feeds deformation
quantification and mechanical prediction, which in turn is the substrate for
a per-part digital twin — but the maturity of each stage differs sharply, and
the stages are rarely integrated end to end in a single published pipeline.
Segmentation (§5) is the most mature stage, with deep-learning methods now
routinely reaching near-98% Dice on curated datasets [29] and foundation
models beginning to close the domain gap for low-annotation settings
[35]; fiber orientation and morphometry (§6) similarly rest on a
well-established formalism (Advani–Tucker [38]) with multiple validated
estimators. Deformation measurement (§8) and mechanical prediction (§9) are
comparatively mature in method but weaker in FRP-specific validation: DVC
adoption for composites is recent [61] and interlaboratory studies show
that measurement uncertainty is scan- and equipment-dependent rather than a
fixed property of the algorithm [60], while ML surrogate models for
property prediction are demonstrated primarily on idealized or narrow
microstructure sets [73, 74]. Digital twins and cross-cutting concerns
(§10–§11) are the least mature: as-manufactured-to-as-designed reconciliation
for composites has been demonstrated mainly at coupon/subcomponent scale
[80], and no reviewed source establishes a full-structure, production-scale
composite digital twin backed by CT-derived, uncertainty-quantified
mechanical prediction.

The following table maps the resulting gaps to this project's open roadmap
epics:

| Gap | Relevant sections | Epic |
| --- | --- | --- |
| Segmentation robustness under low fiber–matrix contrast and cross-scanner/cross-material domain shift | §5, §11 | #18 (segmentation quality) |
| DVC accuracy/uncertainty for CT-derived fiber-composite microstructure specifically, as opposed to bone or granular media | §8, §11 | #19 (DVC) |
| DIC/DVC integration at coupon-to-component scale with quantified, scan-specific uncertainty bounds | §8 | #20 (DIC) |
| As-manufactured-to-as-designed reconciliation and model updating for composite digital twins beyond coupon scale, with an auditable qualification-traceability chain | §10, §11 | #21 (digital twin) |

No single reviewed source spans this entire continuum for FRP composites;
the pipeline this repository implements is, to that extent, an integration
of separately validated stages rather than a reproduction of one established
end-to-end system.

## 13. References

**Standards & qualification**

1. ASTM International. ASTM E1441-19, *Standard Guide for Computed Tomography (CT)*.
2. ASTM International. ASTM E1570-19, *Standard Practice for Fan Beam Computed Tomographic (CT) Examination*.
3. ASTM International. ASTM E1695-20e1, *Standard Test Method for Measurement of Computed Tomography (CT) System Performance*.
4. ASTM International. ASTM E2533-21, *Standard Guide for Nondestructive Examination of Polymer Matrix Composites Used in Aerospace Applications*.
5. International Organization for Standardization. ISO 15708-4:2017, *Non-destructive testing — Radiation methods for computed tomography — Part 4: Qualification*.
6. International Organization for Standardization. ISO 15708-2:2017, *Non-destructive testing — Radiation methods for computed tomography — Part 2: Principles, equipment and samples*.
7. SAE International / CMH-17 Coordination Group. *Composite Materials Handbook (CMH-17), Volume 3: Polymer Matrix Composites Materials Usage, Design, and Analysis*.
8. Verein Deutscher Ingenieure / Verband der Elektrotechnik. VDI/VDE 2630 Blatt 1.2, *Computed tomography in dimensional measurement — Influencing variables on measurement results and recommendations for computed tomography dimensional measurements*.
9. Performance Review Institute. Nadcap AC7114, *Audit Criteria for Nondestructive Testing (NDT)*, applied jointly with NAS 410, *NAS Certification & Qualification of Nondestructive Test Personnel* (Aerospace Industries Association).
10. Naresh, K., Khan, K. A., Umer, R., Cantwell, W. J. (2020). "The use of X-ray computed tomography for design and process modeling of aerospace composites: A review." *Materials & Design*, 190, 108553. DOI: 10.1016/j.matdes.2020.108553
11. Gao, Y., Hu, W., Xin, S., Sun, L. (2022). "A review of applications of CT imaging on fiber reinforced composites." *Journal of Composite Materials*. DOI: 10.1177/00219983211050705
12. Galvez-Hernandez, P., Smith, R., Gaska, K., Mavrogordato, M., Sinclair, I., Kratz, J. (2023). "The effect of X-ray computed tomography scan parameters on porosity assessment of carbon fibre reinforced plastics laminates." *Journal of Composite Materials*. DOI: 10.1177/00219983231209383
13. Nikishkov, Y., Airoldi, L., Makeev, A. (2013). "Measurement of voids in composites by X-ray Computed Tomography." *Composites Science and Technology*, 89, 89–97. DOI: 10.1016/j.compscitech.2013.09.019
14. Elkolali, M., Nogueira, L. P., Rønning, P. O., Alcocer, A. (2022). "Void Content Determination of Carbon Fiber Reinforced Polymers: A Comparison between Destructive and Non-Destructive Methods." *Polymers*, 14(6), 1212. DOI: 10.3390/polym14061212

**Acquisition, thresholding & segmentation**

15. Garcea, S. C., Wang, Y., Withers, P. J. (2018). "X-ray computed tomography of polymer composites." *Composites Science and Technology*, 156, 305–319. DOI: 10.1016/j.compscitech.2017.10.023
16. Münch, B., Trtik, P., Marone, F., Stampanoni, M. (2009). "Stripe and ring artifact removal with combined wavelet–Fourier filtering." *Optics Express*, 17(10), 8567–8591. DOI: 10.1364/OE.17.008567
17. Otsu, N. (1979). "A Threshold Selection Method from Gray-Level Histograms." *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62–66. DOI: 10.1109/TSMC.1979.4310076
18. Liao, P.-S., Chen, T.-S., Chung, P.-C. (2001). "A fast algorithm for multilevel thresholding." *Journal of Information Science and Engineering*, 17(5), 713–727.
19. Emerson, M. J., Jespersen, K. M., Dahl, A. B., Conradsen, K., Mikkelsen, L. P. (2017). "Individual fibre segmentation from 3D X-ray computed tomography for characterising the fibre orientation in unidirectional composite materials." *Composites Part A: Applied Science and Manufacturing*, 97, 83–92. DOI: 10.1016/j.compositesa.2016.12.028
20. Bradley, D., Roth, G. (2007). "Adaptive Thresholding Using the Integral Image." *Journal of Graphics Tools*, 12(2), 13–21. DOI: 10.1080/2151237X.2007.10129236
21. Karamov, R., Breite, C., Lomov, S. V., Sergeichev, I., Swolfs, Y. (2023). "Super-Resolution Processing of Synchrotron CT Images for Automated Fibre Break Analysis of Unidirectional Composites." *Polymers*, 15(9), 2206. DOI: 10.3390/polym15092206
22. Upadhyay, S., George Smith, A., Vandepitte, D., Lomov, S. V., Swolfs, Y., Mehdikhani, M. (2024). "Deep-learning versus greyscale segmentation of voids in X-ray computed tomography images of filament-wound composites." *Composites Part A: Applied Science and Manufacturing*, 177, 107937. DOI: 10.1016/j.compositesa.2023.107937

**Deep learning & foundation-model segmentation**

23. Ronneberger, O., Fischer, P., Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." In *MICCAI 2015*, LNCS 9351, 234–241. DOI: 10.1007/978-3-319-24574-4_28
24. Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., Ronneberger, O. (2016). "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation." In *MICCAI 2016*, LNCS 9901, 424–432. DOI: 10.1007/978-3-319-46723-8_49
25. Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., Maier-Hein, K. H. (2021). "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." *Nature Methods*, 18, 203–211. DOI: 10.1038/s41592-020-01008-z
26. Sinchuk, Y., Kibleur, P., Aelterman, J., Boone, M., Van Paepegem, W. (2020). "Variational and Deep Learning Segmentation of Very-Low-Contrast X-ray Computed Tomography Images of Carbon/Epoxy Woven Composites." *Materials*, 13(4), 936. DOI: 10.3390/ma13040936
27. Badran, A., Marshall, D., Legault, Z., Makovetsky, R., Provencher, B., Piché, N., Marsh, M. (2020). "Automated segmentation of computed tomography images of fiber-reinforced composites by deep learning." *Journal of Materials Science*, 55, 16273–16289. DOI: 10.1007/s10853-020-05148-7
28. Badran, A., Parkinson, D., Ushizima, D., Marshall, D., Maillet, E. (2022). "Validation of Deep Learning Segmentation of CT Images of Fiber-Reinforced Composites." *Journal of Composites Science*, 6(2), 60. DOI: 10.3390/jcs6020060
29. de Siqueira, A. F., Ushizima, D. M., van der Walt, S. J. (2022). "A reusable neural network pipeline for unidirectional fiber segmentation." *Scientific Data*, 9, 32. DOI: 10.1038/s41597-022-01119-6
30. Konopczyński, T., Kröger, T., Zheng, L., Hesser, J. (2019). "Instance Segmentation of Fibers from Low Resolution CT Scans via 3D Deep Embedding Learning." arXiv:1901.01034.
31. Guo, R., Stubbe, J., Zhang, Y., Schlepütz, C. M., Rojas Gomez, C., Mehdikhani, M., Breite, C., Swolfs, Y., Villanueva-Perez, P. (2023). "Deep-learning image enhancement and fibre segmentation from time-resolved computed tomography of fibre-reinforced composites." *Composites Science and Technology*, 244, 110278. DOI: 10.1016/j.compscitech.2023.110278
32. Chen, H., et al. (2023). "Automatic segmentation framework of X-Ray tomography data for multi-phase rock using Swin Transformer approach." *Scientific Data*, 10, 810.
33. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., Girshick, R. (2023). "Segment Anything." In *Proceedings of ICCV 2023*, 4015–4026. arXiv:2304.02643.
34. Wang, H., et al. (2023). "SAM-Med3D: Towards General-purpose Segmentation Models for Volumetric Medical Images." arXiv:2310.15161.
35. Tabassum, A., Ziabari, A. (2025). "Constrained GAN-Generated X-Ray CT Data For Self-Supervised And Foundation-Model Segmentation Of Concrete Microstructures." In *2025 IEEE International Conference on Image Processing (ICIP)*. DOI: 10.1109/ICIP55913.2025.11084399
36. Friemann, J., Mikkelsen, L. P., Oddy, C., Fagerström, M. (2025). "Synthetic, automatically labelled training data for machine learning based X-ray CT image segmentation: Application to 3D-textile carbon fibre reinforced composites." *Composites Part B: Engineering*, 305, 112656. DOI: 10.1016/j.compositesb.2025.112656
37. Yunker, A., Kenesei, P., Sharma, H., Park, J.-S., Miceli, A., Kettimuthu, R. (2026). "Unsupervised Semantic Segmentation in Synchrotron Computed Tomography with Self-Correcting Pseudo Labels." arXiv:2603.00372.

**Fiber orientation, morphometry & tracking**

38. Advani, S. G., Tucker, C. L. III (1987). "The Use of Tensors to Describe and Predict Fiber Orientation in Short Fiber Composites." *Journal of Rheology*, 31(8), 751–784. DOI: 10.1122/1.549945
39. Bay, R. S., Tucker, C. L. III (1992). "Stereological measurement and error estimates for three-dimensional fiber orientation." *Polymer Engineering & Science*, 32(4), 240–253. DOI: 10.1002/pen.760320404
40. Krause, M., Hausherr, J. M., Burgeth, B., Herrmann, C., Krenkel, W. (2010). "Determination of the fibre orientation in composites using the structure tensor and local X-ray transform." *Journal of Materials Science*, 45, 888–896. DOI: 10.1007/s10853-009-4016-4
41. Robb, K., Wirjadi, O., Schladitz, K. (2007). "Fiber Orientation Estimation from 3D Image Data: Practical Algorithms, Visualization, and Interpretation." In *7th International Conference on Hybrid Intelligent Systems (HIS 2007)*. DOI: 10.1109/HIS.2007.26
42. Baranowski, T., Dobrovolskij, D., Dremel, K., Hölzing, A., Lohfink, G., Schladitz, K., Zabler, S. (2019). "Local fiber orientation from X-ray region-of-interest computed tomography of large fiber reinforced composite components." *Composites Science and Technology*, 183, 107786. DOI: 10.1016/j.compscitech.2019.107786
43. Karamov, R., Martulli, L. M., Kerschbaum, M., Sergeichev, I., Swolfs, Y., Lomov, S. V. (2020). "Micro-CT based structure tensor analysis of fibre orientation in random fibre composites versus high-fidelity fibre identification methods." *Composite Structures*, 235, 111818. DOI: 10.1016/j.compstruct.2019.111818
44. Salling, F. B., Jeppesen, N., Sonne, M. R., Hattel, J. H., Mikkelsen, L. P. (2022). "Individual fibre inclination segmentation from X-ray computed tomography using principal component analysis." *Journal of Composite Materials*, 56(1), 47–61. DOI: 10.1177/00219983211052741
45. Zanini, F., Carmignato, S. (2022). "Reference object for traceability establishment in X-ray computed tomography measurements of fiber length in fiber-reinforced polymeric materials." *Precision Engineering*, 77, 133–142. DOI: 10.1016/j.precisioneng.2022.05.003
46. Gomarasca, S., Peeters, D. M. J., Atli-Veltin, B., Dransfeld, C. (2021). "Characterising microstructural organisation in unidirectional composites." *Composites Science and Technology*, 215, 109030. DOI: 10.1016/j.compscitech.2021.109030
47. Julià i Juanola, A., Ruiz i Altisent, M., Coll i Arnau, N., Boada i Oliveras, I. (2023). "A frequency-limited waviness and curvature measurement algorithm for composite fibre trackings." *Measurement*, 206, 112223. DOI: 10.1016/j.measurement.2022.112223
48. Czabaj, M. W., Riccio, M. L., Whitacre, W. W. (2014). "Numerical reconstruction of graphite/epoxy composite microstructure based on sub-micron resolution X-ray computed tomography." *Composites Science and Technology*, 105, 174–182. DOI: 10.1016/j.compscitech.2014.10.017
49. Huang, X., Wen, D., Zhao, Y., Wang, Q., Zhou, W., Deng, D. (2016). "Skeleton-based tracing of curved fibers from 3D X-ray microtomographic imaging." *Results in Physics*, 6, 170–177. DOI: 10.1016/j.rinp.2016.03.008
50. Depriester, D., et al. (2022). "Individual fibre separation in 3D fibrous materials imaged by X-ray tomography." *Journal of Microscopy*, 286(3), 172–187. DOI: 10.1111/jmi.13096

**DVC & DIC**

51. Sutton, M. A., Wolters, W. J., Peters, W. H., Ranson, W. F., McNeill, S. R. (1983). "Determination of displacements using an improved digital correlation method." *Image and Vision Computing*, 1(3), 133–139. DOI: 10.1016/0262-8856(83)90064-1
52. Bay, B. K., Smith, T. S., Fyhrie, D. P., Saad, M. (1999). "Digital volume correlation: Three-dimensional strain mapping using X-ray tomography." *Experimental Mechanics*, 39, 217–226. DOI: 10.1007/BF02323555
53. Hild, F., Roux, S. (2012). "Comparison of Local and Global Approaches to Digital Image Correlation." *Experimental Mechanics*, 52, 1503–1519. DOI: 10.1007/s11340-012-9603-7
54. Buljac, A., Jailin, C., Mendoza, A., Neggers, J., Taillandier-Thomas, T., Bouterf, A., Smaniotto, B., Hild, F., Roux, S. (2018). "Digital Volume Correlation: Review of Progress and Challenges." *Experimental Mechanics*, 58, 661–708. DOI: 10.1007/s11340-018-0390-7
55. Bar-Kochba, E., Toyjanova, J., Andrews, E., Kim, K.-S., Franck, C. (2015). "A Fast Iterative Digital Volume Correlation Algorithm for Large Deformations." *Experimental Mechanics*, 55, 261–274. DOI: 10.1007/s11340-014-9874-2
56. Wong, T. M., Moosmann, J., Zeller-Plumhoff, B. (2024). "VolRAFT: Volumetric Optical Flow Network for Digital Volume Correlation of Synchrotron Radiation-based Micro-CT Images of Bone-Implant Interfaces." In *2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*. DOI: 10.1109/CVPRW63382.2024.00010
57. Stamati, O., Andò, E., Roubin, E., Cailletaud, R., et al. (2020). "spam: Software for Practical Analysis of Materials." *Journal of Open Source Software*, 5(51), 2286. DOI: 10.21105/joss.02286
58. Olufsen, S. N., Andersen, M. E., Fagerholt, E. (2020). "μDIC: An open-source toolkit for digital image correlation." *SoftwareX*, 11, 100391. DOI: 10.1016/j.softx.2019.100391
59. Blaber, J., Adair, B., Antoniou, A. (2015). "Ncorr: Open-Source 2D Digital Image Correlation Matlab Software." *Experimental Mechanics*, 55, 1105–1122. DOI: 10.1007/s11340-015-0009-1
60. Croom, B. P., Burden, D., Jin, H., Vonk, N. H., Hoefnagels, J. P. M., Smaniotto, B., Hild, F., Quintana, E., Sun, Q., Nie, X., Li, X. (2021). "Interlaboratory Study of Digital Volume Correlation Error Due to X-Ray Computed Tomography Equipment and Scan Parameters: an Update from the DVC Challenge." *Experimental Mechanics*, 61, 385–403. DOI: 10.1007/s11340-020-00653-x
61. Mehdikhani, M., Breite, C., Swolfs, Y., Soete, J., Wevers, M., Lomov, S. V., Gorbatikh, L. (2021). "Digital volume correlation for meso/micro in-situ damage analysis in carbon fiber reinforced composites." *Composites Science and Technology*, 213, 108944. DOI: 10.1016/j.compscitech.2021.108944
62. Wang, Y., Chen, Q., Luo, Q., Li, Q., Sun, G. (2024). "Characterizing damage evolution in fiber reinforced composites using in-situ X-ray computed tomography, deep machine learning and digital volume correlation (DVC)." *Composites Science and Technology*, 254, 110650. DOI: 10.1016/j.compscitech.2024.110650
63. Holmes, J., Sommacal, S., Das, R., Stachurski, Z., Compston, P. (2023). "Digital image and volume correlation for deformation and damage characterisation of fibre-reinforced composites: A review." *Composite Structures*, 315, 116994. DOI: 10.1016/j.compstruct.2023.116994
64. Jiang, L., Xiao, S., Wu, J., Liao, Z., Zhu, G., Zhang, B. (2026). "CT and image post-processing for fiber composites: Defect analysis, deep learning, digital volume correlation, and FE simulation – A review." *Materials & Design*, 115698. DOI: 10.1016/j.matdes.2026.115698

**Prediction of mechanical behavior**

65. Eshelby, J. D. (1957). "The determination of the elastic field of an ellipsoidal inclusion, and related problems." *Proceedings of the Royal Society of London A*, 241(1226), 376–396. DOI: 10.1098/rspa.1957.0133
66. Mori, T., Tanaka, K. (1973). "Average stress in matrix and average elastic energy of materials with misfitting inclusions." *Acta Metallurgica*, 21(5), 571–574. DOI: 10.1016/0001-6160(73)90064-3
67. Halpin, J. C., Kardos, J. L. (1976). "The Halpin-Tsai Equations: A Review." *Polymer Engineering & Science*, 16(5), 344–352. DOI: 10.1002/pen.760160512
68. Kanouté, P., Boso, D. P., Chaboche, J. L., Schrefler, B. A. (2009). "Multiscale Methods for Composites: A Review." *Archives of Computational Methods in Engineering*, 16, 31–75. DOI: 10.1007/s11831-008-9028-8
69. Feyel, F., Chaboche, J.-L. (2000). "FE2 multiscale approach for modelling the elastoviscoplastic behaviour of long fibre SiC/Ti composite materials." *Computer Methods in Applied Mechanics and Engineering*, 183(3–4), 309–330. DOI: 10.1016/S0045-7825(99)00224-8
70. Llorca, J., González, C., Molina-Aldareguía, J. M., Segurado, J., Seltzer, R., Sket, F., Rodríguez, M., Sádaba, S., Muñoz, R., Canal, L. P. (2011). "Multiscale Modeling of Composite Materials: a Roadmap Towards Virtual Testing." *Advanced Materials*, 23(44), 5130–5147. DOI: 10.1002/adma.201101683
71. Sinchuk, Y., Shishkina, O., Gueguen, M., Signor, L., Nadot-Martin, C., Trumel, H., Van Paepegem, W. (2022). "X-ray CT based multi-layer unit cell modeling of carbon fiber-reinforced textile composites: Segmentation, meshing and elastic property homogenization." *Composite Structures*, 297, 116003. DOI: 10.1016/j.compstruct.2022.116003
72. Ai, S., Song, W., Chen, Y. (2021). "Stress field and damage evolution in C/SiC woven composites: Image-based finite element analysis and in situ X-ray computed tomography tests." *Journal of the European Ceramic Society*, 41(4), 2482–2491. DOI: 10.1016/j.jeurceramsoc.2020.12.026
73. Sun, Y., Hanhan, I., Sangid, M. D., Lin, G. (2024). "Predicting Mechanical Properties from Microstructure Images in Fiber-Reinforced Polymers Using Convolutional Neural Networks." *Journal of Composites Science*, 8(10), 387. DOI: 10.3390/jcs8100387
74. Caliskan, E., Abedi, R., Lupo Pasini, M. (2025). "Graph neural networks for mechanical property prediction of 2D fiber composites." *Materials & Design*, 114500. DOI: 10.1016/j.matdes.2025.114500
75. Huang, K., Wang, B., Guo, L. (2026). "Application of machine learning in the micromechanics of composites: a review." *Composites Part A: Applied Science and Manufacturing*, 109808. DOI: 10.1016/j.compositesa.2026.109808
76. Kumar C., H., Swamy, R. P. (2021). "Fatigue life prediction of glass fiber reinforced epoxy composites using artificial neural networks." *Composites Communications*, 26, 100812. DOI: 10.1016/j.coco.2021.100812
77. Balokas, G., Kriegesmann, B., Rolfes, R. (2021). "Data-driven inverse uncertainty quantification in the transverse tensile response of carbon fiber reinforced composites." *Composites Science and Technology*, 211, 108845. DOI: 10.1016/j.compscitech.2021.108845
78. García-Merino, J. C., Calvo-Jurado, C., García-Macías, E. (2022). "Polynomial chaos expansion for uncertainty propagation analysis in numerical homogenization of 2D/3D periodic composite microstructures." *Composite Structures*, 300, 116130. DOI: 10.1016/j.compstruct.2022.116130

**Digital twins & cross-cutting concerns**

79. Tuegel, E. J., Ingraffea, A. R., Eason, T. G., Spottswood, S. M. (2011). "Reengineering Aircraft Structural Life Prediction Using a Digital Twin." *International Journal of Aerospace Engineering*, 2011, Article 154798. DOI: 10.1155/2011/154798
80. Hearley, B. L., Pineda, E. J., Bednarcyk, B. A., Baker, J. R., Wilson, L. G. (2026). "Towards the Development of Multiscale Digital Twins for Fiber-Reinforced Composite Materials Using Machine Learning." *Applied Sciences*, 16(8), 3666. DOI: 10.3390/app16083666
81. Ahsan, M. M., Liu, Y., Raman, S., Siddique, Z. (2024). "Digital Twins in Additive Manufacturing: A Systematic Review." arXiv:2409.00877.
82. Bellens, S., Guerrero, P., Vandewalle, P., Dewulf, W. (2024). "Machine learning in industrial X-ray computed tomography – a review." *CIRP Journal of Manufacturing Science and Technology*, 51, 324–341. DOI: 10.1016/j.cirpj.2024.05.004
83. Polevoy, D., Kazimirov, D., Gilmanov, M., Nikolaev, D. (2025). "No Reproducibility, No Progress: Rethinking CT Benchmarking." *Journal of Imaging*, 11(10), 344. DOI: 10.3390/jimaging11100344
