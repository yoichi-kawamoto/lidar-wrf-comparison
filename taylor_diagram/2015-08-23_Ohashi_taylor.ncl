load "./taylor_diagram.ncl"

begin
  title = "2015-08-23_Ohashi"

  var = (/"01d15h", "02d15h", "03d15h", "05d15h", "07d15h", "10d15h", "20d15h", "30d15h", "Average"/)
  nVar = dimsizes(var)

  case = (/"Wind Speed"/)
  nCase = dimsizes(case)

  wks = gsn_open_wks("pdf", title)
  plot = new(3, graphic)

  opt            = True
  opt@Markers    = (/4/)
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
; "Wind Speed"
  CA_ratio   = (/0.9922223433832903,1.0083609258566917,0.8091560003152508,0.7040389675500844,0.7268538491892306,0.6738704540968309,0.707768216313355,0.6201778064865737,0.7005024858325747/)
  CA_cc      = (/0.9109885255683231,0.8668846476850477,0.8413932866724311,0.8332424202562598,0.7691261066586274,0.7617555766052123,0.6520013774627633,0.7953124788583614,0.9051640226146858/)
; arrays to be passed to taylor plot
  ratio      = new((/nCase, nVar/),typeof(CA_cc))
  cc         = new((/nCase, nVar/),typeof(CA_cc))
  ratio(0,:) = CA_ratio
  cc(0,:)    = CA_cc
  opt@Colors = (/"red"/)
  plot(0)    = taylor_diagram(wks, ratio, cc, opt)

  opt@tiMainString = "ERA"
; "Wind Speed"
  CA_ratio   = (/1.1046793913191857,1.1456094970878836,1.0751645790757631,0.9857458148458366,0.9225416601427822,1.0395323009601554,0.951932984042635,0.9663970544965861,0.988922400165466/)
  CA_cc      = (/0.8847225373675893,0.9342543322171882,0.9433188716126644,0.9117209152005044,0.8876040311934107,0.9214122610373161,0.9294791438995198,0.9041719442715942,0.9474775227040901/)
; arrays to be passed to taylor plot
  ratio      = new((/nCase, nVar/),typeof(CA_cc))
  cc         = new((/nCase, nVar/),typeof(CA_cc))
  ratio(0,:) = CA_ratio
  cc(0,:)    = CA_cc
  opt@Colors = (/"blue"/)
  plot(1)    = taylor_diagram(wks, ratio, cc, opt)

  opt@tiMainString = "GPV"
; "Wind Speed"
  CA_ratio   = (/1.0361746677407027,1.084850336110229,1.0151716779233244,0.8364146054931054,0.8455679819665292,0.8382338599916166,0.8255756089779592,0.8275812794436483,0.8667066966822881/)
  CA_cc      = (/0.924047072778043,0.9144678646262846,0.9013909892729925,0.8529510107883232,0.9031155178701957,0.8336233489990337,0.8139269896251116,0.8102965572201292,0.9205801202416507/)
; arrays to be passed to taylor plot
  ratio      = new((/nCase, nVar/),typeof(CA_cc))
  cc         = new((/nCase, nVar/),typeof(CA_cc))
  ratio(0,:) = CA_ratio
  cc(0,:)    = CA_cc
  opt@Colors = (/"green"/)
  plot(2)    = taylor_diagram(wks,ratio,cc,opt)

  resP = True
  resP@gsnPanelMainString = title
  resP@gsnPanelYWhiteSpacePercent = 3

  gsn_panel(wks, plot(0:2), (/1,3/), resP)

end
