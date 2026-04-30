  # UR3 人手交接功能分階段開發計畫

  ## Summary

  目標是把現有「抓取後固定放置」流程，擴充成「依掌心方向選擇接近方式、確認人手抓握後鬆手、退回待機點」的商用級交接功能。第一版策略鎖定為：

  - 感知：MediaPipe Hands 做單手 RGB landmarks 偵測
  - 交接區：固定交接區，不做全域追手
  - 使用情境：單手、多姿態，但人需在限定交接區內
  - 鬆手策略：雙重確認
    先由系統判定「已抓握」，再由明確二次條件才鬆手
  - 安全策略：保守即停
    手消失、方向跳動、離開交接區、超時都立即中止並退回安全位
  - 驗證節奏：先離線資料驗證，再上機械臂

  第一版不追求自然互動感，而是先建立可驗證、可重現、可逐步放寬的安全交接基線。

  ## Implementation Changes

  ### 1. 控制流程重構

  把現有控制器從單一路徑 _grasp_and_place() 改成明確狀態機，至少包含：

  - IDLE
  - PLAN_PREGRASP
  - EXECUTE_PICK
  - LIFT_OBJECT
  - MOVE_TO_HANDOVER_PRE
  - WAIT_HAND_STABLE
  - APPROACH_HANDOVER
  - WAIT_GRASP_CONFIRM
  - WAIT_RELEASE_CONFIRM
  - OPEN_GRIPPER
  - RETREAT_SAFE
  - RETURN_READY
  - ABORT_SAFE

  行為規格：

  - 抓取前半段沿用現有 AnyGrasp + MoveIt 流程
  - 抓取完成後不再直接走固定 final_xyz 放置
  - 進入 handover mode 時，改走「固定交接區 + 掌心方向對應接近位姿」
  - 任一安全條件失敗都進 ABORT_SAFE
  - ABORT_SAFE 不鬆手，先退回安全位，再視情況回待機點

  ### 2. 新增 handover perception 子系統

  新增一個獨立節點，職責只做交接感知，不混進 brain_node.py：

  建議公開介面：

  - 訂閱：RGB 影像
  - 發布：/semantic_handover/hand_state
  - 選配發布：/semantic_handover/debug_image

  hand_state 內容需至少包含：

  - 是否偵測到有效手
  - 手心中心 2D 座標
  - 手心中心對應 3D 點
  - 掌心法向或等價方向向量
  - 手部追蹤穩定度分數
  - 是否位於固定交接區
  - 是否滿足「可接近」條件
  - frame timestamp

  第一版技術細節：

  - 用 MediaPipe Hands 取得 21 點 landmarks
  - 以 palm landmarks 幾何關係估手心中心與掌心法向
  - 用現有深度影像把手心中心投影到 3D
  - 只接受單手主目標
    若多手同時出現，直接標記不安全，不進入接近

  ### 3. 交接位姿與接近策略

  第一版只支援固定交接區，區域以 base frame 定義成 3D box。

  在交接區內，依掌心方向從對應側接近：

  - 若掌心法向滿足正面接物條件，走正向接近
  - 若手掌角度偏左/偏右，在允許範圍內切換到對應側向接近模板
  - 若掌心方向不在可接受角度窗內，不接近，只等待或中止

  第一版不要做連續笛卡兒追手，只做兩段式：

  - 先到 handover_pre_pose
  - 感知連續穩定後，執行短距離 approach_to_hand_pose

  需要定義的靜態參數：

  - 交接區 3D 範圍
  - handover_pre_pose
  - 最大接近距離
  - 最小人手安全距離
  - 掌心方向允許角度窗
  - 方向穩定幀數 / 時間窗
  - approach 速度與加速度上限
  - release 前等待 timeout

  ### 4. 抓握確認與鬆手條件

  第一版用雙重確認，不直接自動鬆手。

  系統判定「已抓握」的建議條件：

  - 手仍在交接區內
  - 掌心與夾爪相對位姿持續穩定一段時間
  - 手指包覆或手心距離變化符合接物趨勢
  - 夾爪 OBJ/POS 狀態未顯示空夾或異常

  鬆手流程：

  - 系統先進入 WAIT_GRASP_CONFIRM
  - 達到系統抓握判定後，進入 WAIT_RELEASE_CONFIRM
  - 第二確認來源第一版建議用明確指令：
    topic、鍵盤命令、或 GUI 按鍵其一，但要唯一且可記錄
  - 收到 release confirm 才張開夾爪
  - 張開後立即後退，再回待機點

  ### 5. ROS / 軟體介面

  第一版可先延用 JSON over std_msgs/String，降低改動面；若後續穩定，再升級自定義 msg。

  至少新增這些介面：

  - /semantic_grasp/task_mode
    {"mode":"place"|"handover"}
  - /semantic_handover/hand_state
  - /semantic_handover/user_cmd
    {"cmd":"release"|"abort"|"resume"}
  - /semantic_handover/state
    對外回報目前狀態機狀態與失敗原因

  client_camera.py 或新 UI 的功能需求：

  - 切換 place / handover
  - 顯示 handover 狀態
  - 顯示是否已進入交接區
  - 顯示是否已達抓握判定
  - 提供 release / abort

  ## Phase Goals And Validation

  ### Phase 0: 規格與資料基線

  目標：

  - 定義交接區、接近模板、狀態機、失敗碼
  - 錄製代表性 RGB-D 交接影片資料集
  - 建立離線回放腳本

  驗證：

  - 至少覆蓋 3 類掌心方向
    正面、左偏、右偏
  - 至少覆蓋正常與失敗場景：
    手進區、手離區、雙手入鏡、掌心亂晃、手背朝向、無手

  交付物：

  - 規格文件
  - 測試資料清單
  - 狀態與失敗碼定義

  ### Phase 1: 離線掌心感知與方向判定

  目標：

  - MediaPipe Hands 穩定輸出單手 landmarks
  - 可從 RGB-D 估手心中心與掌心方向
  - 可把手分類為「可接近 / 不可接近」

  驗證門檻：

  - 離線資料上，手存在判定穩定
  - 掌心方向分類在既定資料集上達到可接受準確率
  - 多手、遮擋、掉幀時能明確回傳不安全，而不是亂輸出接近方向

  接受標準：

  - 每支測試影片都能產出時間對齊的 debug overlay 與狀態 log
  - 判定抖動可被穩定窗抑制
  - 所有 failure case 都有明確 reason code

  ### Phase 2: 固定交接區與接近模板

  目標：

  - 在不碰人的前提下，機械臂可移動到交接前置位
  - 依掌心方向在模板內選對接近姿態
  - 保守即停完整生效

  驗證門檻：

  - 不拿物時，在 mock hand 或靜態手目標前完成接近測試
  - 手離區、方向超窗、追蹤失穩時，機械臂不前進或立即停退
  - 接近過程沒有碰撞桌面、穿越禁區、或超出工作空間

  接受標準：

  - 每種掌心方向模板至少連續成功多次
  - 所有 abort 路徑都回到安全位
  - timeout 與追蹤失敗不會卡死狀態機

  ### Phase 3: 抓物後交接，不鬆手

  目標：

  - 抓取鏈與 handover 鏈打通
  - 物品抓起後能到交接前置位，等待手穩定
  - 還不開放 release，只驗證接近與等待

  驗證門檻：

  - 從真實抓取到 handover 等待的全流程可重現
  - 拿不同尺寸物體時，交接前姿態與安全距離仍合理
  - 感知不穩時系統會保留握持並安全退出

  接受標準：

  - 全流程連續成功多次
  - 沒有誤鬆手
  - 沒有因物體遮擋導致錯誤接近人手

  ### Phase 4: 抓握確認與雙重確認鬆手

  目標：

  - 建立「已抓握」判定
  - 串接 release confirm
  - 鬆手後後退並回待機點

  驗證門檻：

  - 未抓穩時不能進入 release
  - 抓穩後需同時滿足系統判定與第二確認才會放手
  - 放手後固定後退距離與回待機點成功

  接受標準：

  - 無誤放手
  - 無放手後二次碰撞
  - 所有 release 事件、abort 事件、timeout 都有完整 log

  ### Phase 5: 商用品質補強

  目標：
  - 現場可觀測性
  內容：

  - 交接區、方向窗、穩定窗、timeout 全部參數化
  - 加入 session log、關鍵狀態轉移 log、debug overlay 錄影
  - 製作操作 SOP 與故障排除表
  - 建立回歸測試資料集與驗收 checklist

  ## Test Plan

  必做測試案例：

  - 單手進入交接區，掌心正對，成功接近
  - 單手進入交接區，掌心左偏，選對左側接近模板
  - 單手進入交接區，掌心右偏，選對右側接近模板
  - 手在交接區邊界來回晃動，系統不得冒進
  - 手進入後中途離開，系統 abort 並退回
  - 雙手同時出現，系統拒絕接近
  - 手背朝向相機或掌心法向無法可靠估計，系統拒絕接近
  - 抓物後物體遮住部分手掌，系統仍應保守，不可誤判為可放手
  - release confirm 缺失時不得鬆手
  - 鬆手後需固定後退並回待機點
  - 任一失敗後下一次任務可正常重新開始，不殘留狀態

  ## Important Assumptions And Defaults

  - 第一版只做固定交接區，不做全動態追手
  - 第一版只支援單人、單手；多手或多人直接視為不安全
  - 第一版用 MediaPipe Hands，不在 brain_node.py 中整合大型 VLM 流程
  - 第一版先用 JSON over ROS topics；介面穩定後再考慮自定義 msg
  - 第一版 release 必須雙重確認，不做純自動鬆手
  - 第一版以離線資料驗證先行，再進實機驗證
  - 所有感知不確定狀態一律偏向 不接近、不鬆手、先退出

  ## Risks To Discuss Before Implementation

  - 物體本身可能遮住手掌，造成掌心方向不穩，需先定義「遮擋時是否直接禁止交接」
  - RGB hand landmarks 在強光、背光、手套、膚色差異下穩定性可能不足，可能需要預留切換到 RGB-D fusion
  - 固定交接區若設得太小，使用體驗差；太大則安全驗證成本暴增，需要現場量測後定參
  - 夾爪沒有真正的力矩感知，抓握確認不能假裝等同於人已接手，只能做保守推定
  - 若未來要走商業部署，最終還需要補：
    明確急停整合、操作權限、事件追溯、參數鎖版、現場校正流程、以及安全責任邊界
