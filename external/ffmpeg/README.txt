FFmpeg 64-bit static Windows build from www.gyan.dev

Version: 2023-11-22-git-0008e1c5d5-essentials_build-www.gyan.dev

License: GPL v3

Source Code: https://github.com/FFmpeg/FFmpeg/commit/0008e1c5d5

git-essentials build configuration: 

ARCH                      x86 (generic)
big-endian                no
runtime cpu detection     yes
standalone assembly       yes
x86 assembler             nasm
MMX enabled               yes
MMXEXT enabled            yes
3DNow! enabled            yes
3DNow! extended enabled   yes
SSE enabled               yes
SSSE3 enabled             yes
AESNI enabled             yes
AVX enabled               yes
AVX2 enabled              yes
AVX-512 enabled           yes
AVX-512ICL enabled        yes
XOP enabled               yes
FMA3 enabled              yes
FMA4 enabled              yes
i686 features enabled     yes
CMOV is fast              yes
EBX available             yes
EBP available             yes
debug symbols             yes
strip symbols             yes
optimize for size         no
optimizations             yes
static                    yes
shared                    no
postprocessing support    yes
network support           yes
threading support         pthreads
safe bitstream reader     yes
texi2html enabled         no
perl enabled              yes
pod2man enabled           yes
makeinfo enabled          yes
makeinfo supports HTML    yes
xmllint enabled           yes

External libraries:
avisynth                libopencore_amrnb       libvpx
bzlib                   libopencore_amrwb       libwebp
gmp                     libopenjpeg             libx264
gnutls                  libopenmpt              libx265
iconv                   libopus                 libxml2
libaom                  librubberband           libxvid
libass                  libspeex                libzimg
libfontconfig           libsrt                  libzmq
libfreetype             libssh                  lzma
libfribidi              libtheora               mediafoundation
libgme                  libvidstab              sdl2
libgsm                  libvmaf                 zlib
libharfbuzz             libvo_amrwbenc
libmp3lame              libvorbis

External libraries providing hardware acceleration:
amf                     d3d11va                 libvpl
cuda                    dxva2                   nvdec
cuda_llvm               ffnvcodec               nvenc
cuvid                   libmfx

Libraries:
avcodec                 avformat                swresample
avdevice                avutil                  swscale
avfilter                postproc

Programs:
ffmpeg                  ffplay                  ffprobe

Enabled decoders:
aac                     fourxm                  pfm
aac_fixed               fraps                   pgm
aac_latm                frwu                    pgmyuv
aasc                    ftr                     pgssub
ac3                     g2m                     pgx
ac3_fixed               g723_1                  phm
acelp_kelvin            g729                    photocd
adpcm_4xm               gdv                     pictor
adpcm_adx               gem                     pixlet
adpcm_afc               gif                     pjs
adpcm_agm               gremlin_dpcm            png
adpcm_aica              gsm                     ppm
adpcm_argo              gsm_ms                  prores
adpcm_ct                h261                    prosumer
adpcm_dtk               h263                    psd
adpcm_ea                h263i                   ptx
adpcm_ea_maxis_xa       h263p                   qcelp
adpcm_ea_r1             h264                    qdm2
adpcm_ea_r2             h264_cuvid              qdmc
adpcm_ea_r3             h264_qsv                qdraw
adpcm_ea_xas            hap                     qoi
adpcm_g722              hca                     qpeg
adpcm_g726              hcom                    qtrle
adpcm_g726le            hdr                     r10k
adpcm_ima_acorn         hevc                    r210
adpcm_ima_alp           hevc_cuvid              ra_144
adpcm_ima_amv           hevc_qsv                ra_288
adpcm_ima_apc           hnm4_video              ralf
adpcm_ima_apm           hq_hqa                  rasc
adpcm_ima_cunning       hqx                     rawvideo
adpcm_ima_dat4          huffyuv                 realtext
adpcm_ima_dk3           hymt                    rka
adpcm_ima_dk4           iac                     rl2
adpcm_ima_ea_eacs       idcin                   roq
adpcm_ima_ea_sead       idf                     roq_dpcm
adpcm_ima_iss           iff_ilbm                rpza
adpcm_ima_moflex        ilbc                    rscc
adpcm_ima_mtf           imc                     rtv1
adpcm_ima_oki           imm4                    rv10
adpcm_ima_qt            imm5                    rv20
adpcm_ima_rad           indeo2                  rv30
adpcm_ima_smjpeg        indeo3                  rv40
adpcm_ima_ssi           indeo4                  s302m
adpcm_ima_wav           indeo5                  sami
adpcm_ima_ws            interplay_acm           sanm
adpcm_ms                interplay_dpcm          sbc
adpcm_mtaf              interplay_video         scpr
adpcm_psx               ipu                     screenpresso
adpcm_sbpro_2           jacosub                 sdx2_dpcm
adpcm_sbpro_3           jpeg2000                sga
adpcm_sbpro_4           jpegls                  sgi
adpcm_swf               jv                      sgirle
adpcm_thp               kgv1                    sheervideo
adpcm_thp_le            kmvc                    shorten
adpcm_vima              lagarith                simbiosis_imx
adpcm_xa                lead                    sipr
adpcm_xmd               libaom_av1              siren
adpcm_yamaha            libgsm                  smackaud
adpcm_zork              libgsm_ms               smacker
agm                     libopencore_amrnb       smc
aic                     libopencore_amrwb       smvjpeg
alac                    libopus                 snow
alias_pix               libspeex                sol_dpcm
als                     libvorbis               sonic
amrnb                   libvpx_vp8              sp5x
amrwb                   libvpx_vp9              speedhq
amv                     loco                    speex
anm                     lscr                    srgc
ansi                    m101                    srt
anull                   mace3                   ssa
apac                    mace6                   stl
ape                     magicyuv                subrip
apng                    mdec                    subviewer
aptx                    media100                subviewer1
aptx_hd                 metasound               sunrast
arbc                    microdvd                svq1
argo                    mimic                   svq3
ass                     misc4                   tak
asv1                    mjpeg                   targa
asv2                    mjpeg_cuvid             targa_y216
atrac1                  mjpeg_qsv               tdsc
atrac3                  mjpegb                  text
atrac3al                mlp                     theora
atrac3p                 mmvideo                 thp
atrac3pal               mobiclip                tiertexseqvideo
atrac9                  motionpixels            tiff
aura                    movtext                 tmv
aura2                   mp1                     truehd
av1                     mp1float                truemotion1
av1_cuvid               mp2                     truemotion2
av1_qsv                 mp2float                truemotion2rt
avrn                    mp3                     truespeech
avrp                    mp3adu                  tscc
avs                     mp3adufloat             tscc2
avui                    mp3float                tta
ayuv                    mp3on4                  twinvq
bethsoftvid             mp3on4float             txd
bfi                     mpc7                    ulti
bink                    mpc8                    utvideo
binkaudio_dct           mpeg1_cuvid             v210
binkaudio_rdft          mpeg1video              v210x
bintext                 mpeg2_cuvid             v308
bitpacked               mpeg2_qsv               v408
bmp                     mpeg2video              v410
bmv_audio               mpeg4                   vb
bmv_video               mpeg4_cuvid             vble
bonk                    mpegvideo               vbn
brender_pix             mpl2                    vc1
c93                     msa1                    vc1_cuvid
cavs                    mscc                    vc1_qsv
cbd2_dpcm               msmpeg4v1               vc1image
ccaption                msmpeg4v2               vcr1
cdgraphics              msmpeg4v3               vmdaudio
cdtoons                 msnsiren                vmdvideo
cdxl                    msp2                    vmix
cfhd                    msrle                   vmnc
cinepak                 mss1                    vnull
clearvideo              mss2                    vorbis
cljr                    msvideo1                vp3
cllc                    mszh                    vp4
comfortnoise            mts2                    vp5
cook                    mv30                    vp6
cpia                    mvc1                    vp6a
cri                     mvc2                    vp6f
cscd                    mvdv                    vp7
cyuv                    mvha                    vp8
dca                     mwsc                    vp8_cuvid
dds                     mxpeg                   vp8_qsv
derf_dpcm               nellymoser              vp9
dfa                     notchlc                 vp9_cuvid
dfpwm                   nuv                     vp9_qsv
dirac                   on2avc                  vplayer
dnxhd                   opus                    vqa
dolby_e                 osq                     vqc
dpx                     paf_audio               wady_dpcm
dsd_lsbf                paf_video               wavarc
dsd_lsbf_planar         pam                     wavpack
dsd_msbf                pbm                     wbmp
dsd_msbf_planar         pcm_alaw                wcmv
dsicinaudio             pcm_bluray              webp
dsicinvideo             pcm_dvd                 webvtt
dss_sp                  pcm_f16le               wmalossless
dst                     pcm_f24le               wmapro
dvaudio                 pcm_f32be               wmav1
dvbsub                  pcm_f32le               wmav2
dvdsub                  pcm_f64be               wmavoice
dvvideo                 pcm_f64le               wmv1
dxa                     pcm_lxf                 wmv2
dxtory                  pcm_mulaw               wmv3
dxv                     pcm_s16be               wmv3image
eac3                    pcm_s16be_planar        wnv1
eacmv                   pcm_s16le               wrapped_avframe
eamad                   pcm_s16le_planar        ws_snd1
eatgq                   pcm_s24be               xan_dpcm
eatgv                   pcm_s24daud             xan_wc3
eatqi                   pcm_s24le               xan_wc4
eightbps                pcm_s24le_planar        xbin
eightsvx_exp            pcm_s32be               xbm
eightsvx_fib            pcm_s32le               xface
escape124               pcm_s32le_planar        xl
escape130               pcm_s64be               xma1
evrc                    pcm_s64le               xma2
exr                     pcm_s8                  xpm
fastaudio               pcm_s8_planar           xsub
ffv1                    pcm_sga                 xwd
ffvhuff                 pcm_u16be               y41p
ffwavesynth             pcm_u16le               ylc
fic                     pcm_u24be               yop
fits                    pcm_u24le               yuv4
flac                    pcm_u32be               zero12v
flashsv                 pcm_u32le               zerocodec
flashsv2                pcm_u8                  zlib
flic                    pcm_vidc                zmbv
flv                     pcx
fmvc                    pdv

Enabled encoders:
a64multi                hevc_nvenc              pcm_u32be
a64multi5               hevc_qsv                pcm_u32le
aac                     huffyuv                 pcm_u8
aac_mf                  jpeg2000                pcm_vidc
ac3                     jpegls                  pcx
ac3_fixed               libaom_av1              pfm
ac3_mf                  libgsm                  pgm
adpcm_adx               libgsm_ms               pgmyuv
adpcm_argo              libmp3lame              phm
adpcm_g722              libopencore_amrnb       png
adpcm_g726              libopenjpeg             ppm
adpcm_g726le            libopus                 prores
adpcm_ima_alp           libspeex                prores_aw
adpcm_ima_amv           libtheora               prores_ks
adpcm_ima_apm           libvo_amrwbenc          qoi
adpcm_ima_qt            libvorbis               qtrle
adpcm_ima_ssi           libvpx_vp8              r10k
adpcm_ima_wav           libvpx_vp9              r210
adpcm_ima_ws            libwebp                 ra_144
adpcm_ms                libwebp_anim            rawvideo
adpcm_swf               libx264                 roq
adpcm_yamaha            libx264rgb              roq_dpcm
alac                    libx265                 rpza
alias_pix               libxvid                 rv10
amv                     ljpeg                   rv20
anull                   magicyuv                s302m
apng                    mjpeg                   sbc
aptx                    mjpeg_qsv               sgi
aptx_hd                 mlp                     smc
ass                     movtext                 snow
asv1                    mp2                     sonic
asv2                    mp2fixed                sonic_ls
av1_amf                 mp3_mf                  speedhq
av1_nvenc               mpeg1video              srt
av1_qsv                 mpeg2_qsv               ssa
avrp                    mpeg2video              subrip
avui                    mpeg4                   sunrast
ayuv                    msmpeg4v2               svq1
bitpacked               msmpeg4v3               targa
bmp                     msrle                   text
cfhd                    msvideo1                tiff
cinepak                 nellymoser              truehd
cljr                    opus                    tta
comfortnoise            pam                     ttml
dca                     pbm                     utvideo
dfpwm                   pcm_alaw                v210
dnxhd                   pcm_bluray              v308
dpx                     pcm_dvd                 v408
dvbsub                  pcm_f32be               v410
dvdsub                  pcm_f32le               vbn
dvvideo                 pcm_f64be               vc2
eac3                    pcm_f64le               vnull
exr                     pcm_mulaw               vorbis
ffv1                    pcm_s16be               vp9_qsv
ffvhuff                 pcm_s16be_planar        wavpack
fits                    pcm_s16le               wbmp
flac                    pcm_s16le_planar        webvtt
flashsv                 pcm_s24be               wmav1
flashsv2                pcm_s24daud             wmav2
flv                     pcm_s24le               wmv1
g723_1                  pcm_s24le_planar        wmv2
gif                     pcm_s32be               wrapped_avframe
h261                    pcm_s32le               xbm
h263                    pcm_s32le_planar        xface
h263p                   pcm_s64be               xsub
h264_amf                pcm_s64le               xwd
h264_mf                 pcm_s8                  y41p
h264_nvenc              pcm_s8_planar           yuv4
h264_qsv                pcm_u16be               zlib
hdr                     pcm_u16le               zmbv
hevc_amf                pcm_u24be
hevc_mf                 pcm_u24le

Enabled hwaccels:
av1_d3d11va             hevc_nvdec              vc1_nvdec
av1_d3d11va2            mjpeg_nvdec             vp8_nvdec
av1_dxva2               mpeg1_nvdec             vp9_d3d11va
av1_nvdec               mpeg2_d3d11va           vp9_d3d11va2
h264_d3d11va            mpeg2_d3d11va2          vp9_dxva2
h264_d3d11va2           mpeg2_dxva2             vp9_nvdec
h264_dxva2              mpeg2_nvdec             wmv3_d3d11va
h264_nvdec              mpeg4_nvdec             wmv3_d3d11va2
hevc_d3d11va            vc1_d3d11va             wmv3_dxva2
hevc_d3d11va2           vc1_d3d11va2            wmv3_nvdec
hevc_dxva2              vc1_dxva2

Enabled parsers:
aac                     dvdsub                  mpegaudio
aac_latm                evc                     mpegvideo
ac3                     flac                    opus
adx                     ftr                     png
amr                     g723_1                  pnm
av1                     g729                    qoi
avs2                    gif                     rv34
avs3                    gsm                     sbc
bmp                     h261                    sipr
cavsvideo               h263                    tak
cook                    h264                    vc1
cri                     hdr                     vorbis
dca                     hevc                    vp3
dirac                   ipu                     vp8
dnxhd                   jpeg2000                vp9
dolby_e                 jpegxl                  vvc
dpx                     misc4                   webp
dvaudio                 mjpeg                   xbm
dvbsub                  mlp                     xma
dvd_nav                 mpeg4video              xwd

Enabled demuxers:
aa                      idf                     pcm_mulaw
aac                     iff                     pcm_s16be
aax                     ifv                     pcm_s16le
ac3                     ilbc                    pcm_s24be
ac4                     image2                  pcm_s24le
ace                     image2_alias_pix        pcm_s32be
acm                     image2_brender_pix      pcm_s32le
act                     image2pipe              pcm_s8
adf                     image_bmp_pipe          pcm_u16be
adp                     image_cri_pipe          pcm_u16le
ads                     image_dds_pipe          pcm_u24be
adx                     image_dpx_pipe          pcm_u24le
aea                     image_exr_pipe          pcm_u32be
afc                     image_gem_pipe          pcm_u32le
aiff                    image_gif_pipe          pcm_u8
aix                     image_hdr_pipe          pcm_vidc
alp                     image_j2k_pipe          pdv
amr                     image_jpeg_pipe         pjs
amrnb                   image_jpegls_pipe       pmp
amrwb                   image_jpegxl_pipe       pp_bnk
anm                     image_pam_pipe          pva
apac                    image_pbm_pipe          pvf
apc                     image_pcx_pipe          qcp
ape                     image_pfm_pipe          r3d
apm                     image_pgm_pipe          rawvideo
apng                    image_pgmyuv_pipe       realtext
aptx                    image_pgx_pipe          redspark
aptx_hd                 image_phm_pipe          rka
aqtitle                 image_photocd_pipe      rl2
argo_asf                image_pictor_pipe       rm
argo_brp                image_png_pipe          roq
argo_cvg                image_ppm_pipe          rpl
asf                     image_psd_pipe          rsd
asf_o                   image_qdraw_pipe        rso
ass                     image_qoi_pipe          rtp
ast                     image_sgi_pipe          rtsp
au                      image_sunrast_pipe      s337m
av1                     image_svg_pipe          sami
avi                     image_tiff_pipe         sap
avisynth                image_vbn_pipe          sbc
avr                     image_webp_pipe         sbg
avs                     image_xbm_pipe          scc
avs2                    image_xpm_pipe          scd
avs3                    image_xwd_pipe          sdns
bethsoftvid             imf                     sdp
bfi                     ingenient               sdr2
bfstm                   ipmovie                 sds
bink                    ipu                     sdx
binka                   ircam                   segafilm
bintext                 iss                     ser
bit                     iv8                     sga
bitpacked               ivf                     shorten
bmv                     ivr                     siff
boa                     jacosub                 simbiosis_imx
bonk                    jpegxl_anim             sln
brstm                   jv                      smacker
c93                     kux                     smjpeg
caf                     kvag                    smush
cavsvideo               laf                     sol
cdg                     libgme                  sox
cdxl                    libopenmpt              spdif
cine                    live_flv                srt
codec2                  lmlm4                   stl
codec2raw               loas                    str
concat                  lrc                     subviewer
dash                    luodat                  subviewer1
data                    lvf                     sup
daud                    lxf                     svag
dcstr                   m4v                     svs
derf                    matroska                swf
dfa                     mca                     tak
dfpwm                   mcc                     tedcaptions
dhav                    mgsts                   thp
dirac                   microdvd                threedostr
dnxhd                   mjpeg                   tiertexseq
dsf                     mjpeg_2000              tmv
dsicin                  mlp                     truehd
dss                     mlv                     tta
dts                     mm                      tty
dtshd                   mmf                     txd
dv                      mods                    ty
dvbsub                  moflex                  usm
dvbtxt                  mov                     v210
dxa                     mp3                     v210x
ea                      mpc                     vag
ea_cdata                mpc8                    vc1
eac3                    mpegps                  vc1t
epaf                    mpegts                  vividas
evc                     mpegtsraw               vivo
ffmetadata              mpegvideo               vmd
filmstrip               mpjpeg                  vobsub
fits                    mpl2                    voc
flac                    mpsub                   vpk
flic                    msf                     vplayer
flv                     msnwc_tcp               vqf
fourxm                  msp                     vvc
frm                     mtaf                    w64
fsb                     mtv                     wady
fwse                    musx                    wav
g722                    mv                      wavarc
g723_1                  mvi                     wc3
g726                    mxf                     webm_dash_manifest
g726le                  mxg                     webvtt
g729                    nc                      wsaud
gdv                     nistsphere              wsd
genh                    nsp                     wsvqa
gif                     nsv                     wtv
gsm                     nut                     wv
gxf                     nuv                     wve
h261                    obu                     xa
h263                    ogg                     xbin
h264                    oma                     xmd
hca                     osq                     xmv
hcom                    paf                     xvag
hevc                    pcm_alaw                xwma
hls                     pcm_f32be               yop
hnm                     pcm_f32le               yuv4mpegpipe
ico                     pcm_f64be
idcin                   pcm_f64le

Enabled muxers:
a64                     h261                    pcm_s16le
ac3                     h263                    pcm_s24be
ac4                     h264                    pcm_s24le
adts                    hash                    pcm_s32be
adx                     hds                     pcm_s32le
aiff                    hevc                    pcm_s8
alp                     hls                     pcm_u16be
amr                     ico                     pcm_u16le
amv                     ilbc                    pcm_u24be
apm                     image2                  pcm_u24le
apng                    image2pipe              pcm_u32be
aptx                    ipod                    pcm_u32le
aptx_hd                 ircam                   pcm_u8
argo_asf                ismv                    pcm_vidc
argo_cvg                ivf                     psp
asf                     jacosub                 rawvideo
asf_stream              kvag                    rm
ass                     latm                    roq
ast                     lrc                     rso
au                      m4v                     rtp
avi                     matroska                rtp_mpegts
avif                    matroska_audio          rtsp
avm2                    md5                     sap
avs2                    microdvd                sbc
avs3                    mjpeg                   scc
bit                     mkvtimestamp_v2         segafilm
caf                     mlp                     segment
cavsvideo               mmf                     smjpeg
codec2                  mov                     smoothstreaming
codec2raw               mp2                     sox
crc                     mp3                     spdif
dash                    mp4                     spx
data                    mpeg1system             srt
daud                    mpeg1vcd                stream_segment
dfpwm                   mpeg1video              streamhash
dirac                   mpeg2dvd                sup
dnxhd                   mpeg2svcd               swf
dts                     mpeg2video              tee
dv                      mpeg2vob                tg2
eac3                    mpegts                  tgp
evc                     mpjpeg                  truehd
f4v                     mxf                     tta
ffmetadata              mxf_d10                 ttml
fifo                    mxf_opatom              uncodedframecrc
fifo_test               null                    vc1
filmstrip               nut                     vc1t
fits                    obu                     voc
flac                    oga                     vvc
flv                     ogg                     w64
framecrc                ogv                     wav
framehash               oma                     webm
framemd5                opus                    webm_chunk
g722                    pcm_alaw                webm_dash_manifest
g723_1                  pcm_f32be               webp
g726                    pcm_f32le               webvtt
g726le                  pcm_f64be               wsaud
gif                     pcm_f64le               wtv
gsm                     pcm_mulaw               wv
gxf                     pcm_s16be               yuv4mpegpipe

Enabled protocols:
async                   http                    rtmp
cache                   httpproxy               rtmpe
concat                  https                   rtmps
concatf                 icecast                 rtmpt
crypto                  ipfs_gateway            rtmpte
data                    ipns_gateway            rtmpts
fd                      libsrt                  rtp
ffrtmpcrypt             libssh                  srtp
ffrtmphttp              libzmq                  subfile
file                    md5                     tcp
ftp                     mmsh                    tee
gopher                  mmst                    tls
gophers                 pipe                    udp
hls                     prompeg                 udplite

Enabled filters:
a3dscope                crossfeed               owdenoise
abench                  crystalizer             pad
abitscope               cue                     pal100bars
acompressor             curves                  pal75bars
acontrast               datascope               palettegen
acopy                   dblur                   paletteuse
acrossfade              dcshift                 pan
acrossover              dctdnoiz                perms
acrusher                ddagrab                 perspective
acue                    deband                  phase
addroi                  deblock                 photosensitivity
adeclick                decimate                pixdesctest
adeclip                 deconvolve              pixelize
adecorrelate            dedot                   pixscope
adelay                  deesser                 pp
adenorm                 deflate                 pp7
aderivative             deflicker               premultiply
adrawgraph              deinterlace_qsv         prewitt
adrc                    dejudder                pseudocolor
adynamicequalizer       delogo                  psnr
adynamicsmooth          derain                  pullup
aecho                   deshake                 qp
aemphasis               despill                 random
aeval                   detelecine              readeia608
aevalsrc                dialoguenhance          readvitc
aexciter                dilation                realtime
afade                   displace                remap
afdelaysrc              dnn_classify            removegrain
afftdn                  dnn_detect              removelogo
afftfilt                dnn_processing          repeatfields
afifo                   doubleweave             replaygain
afir                    drawbox                 reverse
afireqsrc               drawgraph               rgbashift
afirsrc                 drawgrid                rgbtestsrc
aformat                 drawtext                roberts
afreqshift              drmeter                 rotate
afwtdn                  dynaudnorm              rubberband
agate                   earwax                  sab
agraphmonitor           ebur128                 scale
ahistogram              edgedetect              scale2ref
aiir                    elbg                    scale_cuda
aintegral               entropy                 scale_qsv
ainterleave             epx                     scdet
alatency                eq                      scharr
alimiter                equalizer               scroll
allpass                 erosion                 segment
allrgb                  estdif                  select
allyuv                  exposure                selectivecolor
aloop                   extractplanes           sendcmd
alphaextract            extrastereo             separatefields
alphamerge              fade                    setdar
amerge                  feedback                setfield
ametadata               fftdnoiz                setparams
amix                    fftfilt                 setpts
amovie                  field                   setrange
amplify                 fieldhint               setsar
amultiply               fieldmatch              settb
anequalizer             fieldorder              shear
anlmdn                  fifo                    showcqt
anlmf                   fillborders             showcwt
anlms                   find_rect               showfreqs
anoisesrc               firequalizer            showinfo
anull                   flanger                 showpalette
anullsink               floodfill               showspatial
anullsrc                format                  showspectrum
apad                    fps                     showspectrumpic
aperms                  framepack               showvolume
aphasemeter             framerate               showwaves
aphaser                 framestep               showwavespic
aphaseshift             freezedetect            shuffleframes
apsnr                   freezeframes            shufflepixels
apsyclip                fspp                    shuffleplanes
apulsator               gblur                   sidechaincompress
arealtime               geq                     sidechaingate
aresample               gradfun                 sidedata
areverse                gradients               sierpinski
arls                    graphmonitor            signalstats
arnndn                  grayworld               signature
asdr                    greyedge                silencedetect
asegment                guided                  silenceremove
aselect                 haas                    sinc
asendcmd                haldclut                sine
asetnsamples            haldclutsrc             siti
asetpts                 hdcd                    smartblur
asetrate                headphone               smptebars
asettb                  hflip                   smptehdbars
ashowinfo               highpass                sobel
asidedata               highshelf               spectrumsynth
asisdr                  hilbert                 speechnorm
asoftclip               histeq                  split
aspectralstats          histogram               spp
asplit                  hqdn3d                  sr
ass                     hqx                     ssim
astats                  hstack                  ssim360
astreamselect           hstack_qsv              stereo3d
asubboost               hsvhold                 stereotools
asubcut                 hsvkey                  stereowiden
asupercut               hue                     streamselect
asuperpass              huesaturation           subtitles
asuperstop              hwdownload              super2xsai
atadenoise              hwmap                   superequalizer
atempo                  hwupload                surround
atilt                   hwupload_cuda           swaprect
atrim                   hysteresis              swapuv
avectorscope            identity                tblend
avgblur                 idet                    telecine
avsynctest              il                      testsrc
axcorrelate             inflate                 testsrc2
azmq                    interlace               thistogram
backgroundkey           interleave              threshold
bandpass                join                    thumbnail
bandreject              kerndeint               thumbnail_cuda
bass                    kirsch                  tile
bbox                    lagfun                  tiltshelf
bench                   latency                 tinterlace
bilateral               lenscorrection          tlut2
bilateral_cuda          libvmaf                 tmedian
biquad                  life                    tmidequalizer
bitplanenoise           limitdiff               tmix
blackdetect             limiter                 tonemap
blackframe              loop                    tpad
blend                   loudnorm                transpose
blockdetect             lowpass                 treble
blurdetect              lowshelf                tremolo
bm3d                    lumakey                 trim
boxblur                 lut                     unpremultiply
bwdif                   lut1d                   unsharp
bwdif_cuda              lut2                    untile
cas                     lut3d                   uspp
ccrepack                lutrgb                  v360
cellauto                lutyuv                  vaguedenoiser
channelmap              mandelbrot              varblur
channelsplit            maskedclamp             vectorscope
chorus                  maskedmax               vflip
chromahold              maskedmerge             vfrdet
chromakey               maskedmin               vibrance
chromakey_cuda          maskedthreshold         vibrato
chromanr                maskfun                 vidstabdetect
chromashift             mcdeint                 vidstabtransform
ciescope                mcompand                vif
codecview               median                  vignette
color                   mergeplanes             virtualbass
colorbalance            mestimate               vmafmotion
colorchannelmixer       metadata                volume
colorchart              midequalizer            volumedetect
colorcontrast           minterpolate            vpp_qsv
colorcorrect            mix                     vstack
colorhold               monochrome              vstack_qsv
colorize                morpho                  w3fdif
colorkey                movie                   waveform
colorlevels             mpdecimate              weave
colormap                mptestsrc               xbr
colormatrix             msad                    xcorrelate
colorspace              multiply                xfade
colorspace_cuda         negate                  xmedian
colorspectrum           nlmeans                 xstack
colortemperature        nnedi                   xstack_qsv
compand                 noformat                yadif
compensationdelay       noise                   yadif_cuda
concat                  normalize               yaepblur
convolution             null                    yuvtestsrc
convolve                nullsink                zmq
copy                    nullsrc                 zoneplate
corr                    oscilloscope            zoompan
cover_rect              overlay                 zscale
crop                    overlay_cuda
cropdetect              overlay_qsv

Enabled bsfs:
aac_adtstoasc           h264_redundant_pps      pcm_rechunk
av1_frame_merge         hapqa_extract           pgs_frame_merge
av1_frame_split         hevc_metadata           prores_metadata
av1_metadata            hevc_mp4toannexb        remove_extradata
chomp                   imx_dump_header         setts
dca_core                media100_to_mjpegb      text2movsub
dts2pts                 mjpeg2jpeg              trace_headers
dump_extradata          mjpega_dump_header      truehd_core
dv_error_marker         mov2textsub             vp9_metadata
eac3_core               mp3_header_decompress   vp9_raw_reorder
evc_frame_merge         mpeg2_metadata          vp9_superframe
extract_extradata       mpeg4_unpack_bframes    vp9_superframe_split
filter_units            noise                   vvc_metadata
h264_metadata           null                    vvc_mp4toannexb
h264_mp4toannexb        opus_metadata

Enabled indevs:
dshow                   lavfi
gdigrab                 vfwcap

Enabled outdevs:
sdl2

git-essentials external libraries' versions: 

AMF v1.4.30-2-g2f32635
aom v3.7.0-659-g690762d1cb
AviSynthPlus v3.7.3-26-gd2b7c666
ffnvcodec n12.1.14.0-1-g75f032b
freetype VER-2-13-2
fribidi v1.0.13
gsm 1.0.22
harfbuzz 8.3.0-9-gdf635ab78
lame 3.100
libass 0.17.0-61-g9f4e6af
libgme 0.6.3
libopencore-amrnb 0.1.6
libopencore-amrwb 0.1.6
libssh 0.10.4
libtheora 1.1.1
libwebp v1.3.2-105-g34c80749
oneVPL 2.9
openmpt libopenmpt-0.6.12-41-g94865a3dc
opus v1.4-9-gc8549975
rubberband v1.8.1
SDL release-2.28.0-253-g1bb5448fe
speex Speex-1.2.1-20-g3693431
srt v1.5.3-17-g4a8067c
vidstab v1.1.1-4-g05829db
vmaf v2.3.1-178-g013e3234
vo-amrwbenc 0.1.3
vorbis v1.3.7-10-g84c02369
vpx v1.13.1-530-g9142314c2
x264 v0.164.3164
x265 3.5-151-g8ee01d45b
xvid v1.3.7
zeromq 4.3.5
zimg release-3.0.5-150-g7143181

