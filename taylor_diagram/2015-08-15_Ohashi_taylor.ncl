load "./taylor_diagram.ncl"

begin
  title = "2015-08-15_Ohashi"

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
  CA_ratio   = (/0.9630985857027722,0.9507063486839455,1.2109947991850336,1.0889988824882213,1.3919385246842952,1.4171647800067422,1.2935840806127166,1.3205440845501117,0.9426700698547109/)
  CA_cc      = (/0.6855529838698619,0.7180305061627351,0.1966046713257003,0.47826422975192795,0.27752403808013093,0.26844174613726346,0.36570489865283223,0.3654648207872246,0.507098515825551/)
; arrays to be passed to taylor plot
  ratio      = new((/nCase, nVar/),typeof(CA_cc))
  cc         = new((/nCase, nVar/),typeof(CA_cc))
  ratio(0,:) = CA_ratio
  cc(0,:)    = CA_cc
  opt@Colors = (/"red"/)
  plot(0)    = taylor_diagram(wks, ratio, cc, opt)

  opt@tiMainString = "ERA"
; "Wind Speed"
  CA_ratio   = (/1.1314644264480471,1.4720634046464152,0.9102206633361486,1.049463775146117,1.0349422135737567,1.0115569402508846,1.061973649969107,1.054164893546135,0.8739112929183583/)
  CA_cc      = (/0.72589474882568,0.6768667291087471,0.4725556526640378,0.5138471892675763,0.48887547271582377,0.5025522288196664,0.3752037054484738,0.4954698344003744,0.6754390017853894/)
; arrays to be passed to taylor plot
  ratio      = new((/nCase, nVar/),typeof(CA_cc))
  cc         = new((/nCase, nVar/),typeof(CA_cc))
  ratio(0,:) = CA_ratio
  cc(0,:)    = CA_cc
  opt@Colors = (/"blue"/)
  plot(1)    = taylor_diagram(wks, ratio, cc, opt)

  opt@tiMainString = "GPV"
; "Wind Speed"
  CA_ratio   = (/0.9414475017060636,1.1369265756187692,1.1158620983358372,1.0156271833837522,1.185219969999184,0.9185975972076972,1.136171129878187,1.1747211981921415,0.8668094254912224/)
  CA_cc      = (/0.5332428075499879,0.668491952360732,0.6085088775287577,0.6967453566070486,0.4270860851409634,0.26905169186223893,0.3800939477818315,0.41753001300978193,0.6236039443892631/)
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
