load "./taylor_diagram.ncl"

begin
  title = "2015-08-23_Fukuoka"

  var = (/"01d15h", "02d15h", "03d15h", "05d15h", "07d15h", "10d15h", "20d15h", "30d15h"/)
  nVar = dimsizes(var)

  case = (/"Sea Level Pressure", "Temperature", "Wind Speed"/)
  nCase = dimsizes(case)

  wks = gsn_open_wks("pdf", title)
  plot = new(3, graphic)

  opt            = True
  opt@Markers    = (/4, 7, 6/)
  opt@gsMarkerSizeF = 0.006

  ; opt@varLabels = var
  opt@caseLabels = case

  opt@varLabelsYloc = 0.65
  opt@varLabelsFontHeightF = 0.02
  opt@caseLabelsFontHeightF = 0.1

  opt@stnRad = (/0.5, 1.5/)
  opt@ccRays = (/0.6, 0.9/)

  opt@centerDiffRMS = True

  opt@taylorDraw = False
  opt@taylorFrame = False

  opt@tiMainString = "FNL"
; "Sea Level Pressure"
  CA_ratio   = (/0.8788821363913819,1.0261312817704156,1.0843537440583735,1.1586285561039424,1.2155351548872984,1.1885027822886356,1.2696963182195655,1.1859259212672832/)
  CA_cc      = (/0.7960082366760145,0.7803511649879088,0.6691820425811088,0.7143199118454946,0.6922158655534154,0.7057331779886671,0.7254052814591023,0.7150539238392769/)
; "Temperature"
  CB_ratio   = (/1.0399927376122176,0.8612400479773683,0.6904852094203485,0.6276233064285766,0.6669635051924254,0.6528423543392923,0.6751087598488353,0.6048551667619477/)
  CB_cc      = (/0.9817859724069534,0.9832670872217667,0.9738268386129689,0.9019715151904699,0.9738622905099744,0.9819354191787762,0.9789013393347695,0.9736207172883561/)
; "Wind Speed"
  CC_ratio   = (/1.2149222178126224,1.3668824304101284,1.1525567384540123,0.9967964821083711,0.9198693457723129,0.9638749371336385,0.8971419882193646,1.0470180914050768/)
  CC_cc      = (/0.930039248461503,0.9311412311711242,0.9266253025707595,0.9444925600136899,0.8987662600939228,0.9275143848832083,0.8698979873738445,0.9314471506122134/)
; arrays to be passed to taylor plot
  ratio      = new((/nCase, nVar/),typeof(CA_cc))
  cc         = new((/nCase, nVar/),typeof(CA_cc))
  ratio(0,:) = CA_ratio
  ratio(1,:) = CB_ratio
  ratio(2,:) = CC_ratio
  cc(0,:)    = CA_cc
  cc(1,:)    = CB_cc
  cc(2,:)    = CC_cc
  opt@Colors = (/"red", "red", "red"/)
  plot(0)    = taylor_diagram(wks, ratio, cc, opt)

  opt@tiMainString = "ERA"
; "Sea Level Pressure"
  CA_ratio   = (/1.3387907810897706,1.0685126489716694,1.0133789127517845,1.1483640147881713,1.0445921662521986,1.1155945219161614,1.101476109928214,1.1015815187294598/)
  CA_cc      = (/0.7169704382096825,0.8382258391806863,0.8653011707085296,0.8411752298959699,0.8546014069416954,0.8550606358433792,0.8641162979118016,0.8737365512849159/)
; "Temperature"
  CB_ratio   = (/0.990561625114483,0.8500798042759146,0.8341530220905614,0.6793930122561642,0.5662723120323903,0.6389640074361714,0.6293121869677585,0.6217180789789916/)
  CB_cc      = (/0.9417160210438971,0.9855887784456114,0.9889532278799328,0.9836412439491158,0.9732289473552811,0.9844180627223708,0.9834515034168912,0.9828721736540645/)
; "Wind Speed"
  CC_ratio   = (/1.3352770478115594,1.2063836901733724,1.0409656635008195,1.2375366386024245,1.179690528868697,1.1682621243961653,1.1680051336462338,1.217762558370178/)
  CC_cc      = (/0.9268655148672914,0.943805238312513,0.9394594703206741,0.9336958204024273,0.9060249267049237,0.90809199172599,0.900254536528076,0.9230936729661974/)
; arrays to be passed to taylor plot
  ratio      = new((/nCase, nVar/),typeof(CA_cc))
  cc         = new((/nCase, nVar/),typeof(CA_cc))
  ratio(0,:) = CA_ratio
  ratio(1,:) = CB_ratio
  ratio(2,:) = CC_ratio
  cc(0,:)    = CA_cc
  cc(1,:)    = CB_cc
  cc(2,:)    = CC_cc
  opt@Colors = (/"blue", "blue", "blue"/)
  plot(1)    = taylor_diagram(wks, ratio, cc, opt)

  opt@tiMainString = "GPV"
; "Sea Level Pressure"
  CA_ratio   = (/0.8372789629067711,1.099639358935229,1.1908027248539441,1.1005359742308523,1.0723300214599918,1.0018761054081822,1.055076689851601,1.0105876839393444/)
  CA_cc      = (/0.7221785712284328,0.7901246458602703,0.7721137606018739,0.7689913601608585,0.7967107826865273,0.7782835725242874,0.7987358854739269,0.7848037352567352/)
; "Temperature"
  CB_ratio   = (/0.9910725372800402,0.8531181093792458,0.7500080548696202,0.7393556066918974,0.7213402409174752,0.761742525702635,0.7507398850110696,0.6129478259226964/)
  CB_cc      = (/0.9837157235165677,0.9856279450139551,0.9829764915690609,0.979920557152598,0.9853665093061746,0.9818254976391775,0.9889811917996534,0.9802480009880792/)
; "Wind Speed"
  CC_ratio   = (/1.2733499874280003,1.1872788857954222,1.0515846713561947,1.0206386950612576,0.9832168102978148,0.8822934067681827,0.8514890846053078,1.006700087696306/)
  CC_cc      = (/0.9481635599981538,0.9430909500727411,0.9306691000789993,0.940546401414118,0.9462328366850613,0.903479485872011,0.8901268945700488,0.9346590475804174/)
; arrays to be passed to taylor plot
  ratio      = new((/nCase, nVar/),typeof(CA_cc))
  cc         = new((/nCase, nVar/),typeof(CA_cc))
  ratio(0,:) = CA_ratio
  ratio(1,:) = CB_ratio
  ratio(2,:) = CC_ratio
  cc(0,:)    = CA_cc
  cc(1,:)    = CB_cc
  cc(2,:)    = CC_cc
  opt@Colors = (/"green", "green", "green"/)
  plot(2)    = taylor_diagram(wks,ratio,cc,opt)

  resP = True
  resP@gsnPanelMainString = title
  resP@gsnPanelYWhiteSpacePercent = 3

  gsn_panel(wks, plot(0:2), (/1,3/), resP)

end