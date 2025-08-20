# Pull Request · feature/crop-cassava-minimal → main
# Title: Enable cassava (`cas`) in ISIMIP→FAO crop workflow (minimal change)

## Summary
- Add `cas` to ISIMIP crop codes (validation).
- Map `cas` → FAO item "Cassava".
- Update tutorial param lists.
- Add a tiny unit test (mappings + validator).
No impact/logic changes. No new dependencies.

## Files changed (3)

### 1) climada_petals/hazard/relative_cropyield.py
diff --git a/climada_petals/hazard/relative_cropyield.py b/climada_petals/hazard/relative_cropyield.py
index 2bb2abc..7c1e9f4 100644
--- a/climada_petals/hazard/relative_cropyield.py
+++ b/climada_petals/hazard/relative_cropyield.py
@@
 ISIMIP_CROP_CODES = {
     "whe": "wheat",
     "mai": "maize",
     "soy": "soy",
     "ric": "rice",
+    # add cassava (ISIMIP short code)
+    "cas": "cassava",
 }
@@
 def _check_crop_code(crop: str) -> str:
     crop = crop.lower()
     if crop not in ISIMIP_CROP_CODES:
         raise ValueError(f"Unknown crop code '{crop}'. Allowed: {sorted(ISIMIP_CROP_CODES.keys())}")
     return crop

### 2) climada_petals/entity/exposures/crop_production.py
diff --git a/climada_petals/entity/exposures/crop_production.py b/climada_petals/entity/exposures/crop_production.py
index 1a2f9d1..a1f3b7a 100644
--- a/climada_petals/entity/exposures/crop_production.py
+++ b/climada_petals/entity/exposures/crop_production.py
@@ CROP2FAO = {
     "whe": "Wheat",
     "mai": "Maize",
     "soy": "Soybeans",
     "ric": "Rice, paddy",
+    "cas": "Cassava",
 }
@@ CROP_PRETTY = {
     "whe": "wheat",
     "mai": "maize",
     "soy": "soy",
     "ric": "rice (paddy)",
+    "cas": "cassava",
 }
@@ class CropProduction(Exposures):
         if crop not in CROP2FAO:
             raise ValueError(
                 f"Unknown crop '{crop}'. Supported: {sorted(CROP2FAO.keys())}"
             )
         self.crop = crop

### 3) doc/tutorial/climada_hazard_entity_Crop.rst
diff --git a/doc/tutorial/climada_hazard_entity_Crop.rst b/doc/tutorial/climada_hazard_entity_Crop.rst
index e4a5d22..3f77aca 100644
--- a/doc/tutorial/climada_hazard_entity_Crop.rst
+++ b/doc/tutorial/climada_hazard_entity_Crop.rst
@@
-    crop (str): crop type, e.g. 'whe', 'mai', 'soy' or 'ric'
+    crop (str): crop type, e.g. 'whe', 'mai', 'soy', 'ric' or 'cas'
@@
-    crop (string): crop type, e.g. 'mai', 'ric', 'whe', 'soy'
+    crop (string): crop type, e.g. 'mai', 'ric', 'whe', 'soy', 'cas'

## New tests (1)

### tests/test_cassava_minimal.py
+import pytest
+from climada_petals.hazard.relative_cropyield import _check_crop_code, ISIMIP_CROP_CODES
+from climada_petals.entity.exposures.crop_production import CROP2FAO, CROP_PRETTY
+
+def test_cassava_mappings_exist():
+    assert "cas" in ISIMIP_CROP_CODES and ISIMIP_CROP_CODES["cas"] == "cassava"
+    assert CROP2FAO["cas"] == "Cassava"
+    assert CROP_PRETTY["cas"] == "cassava"
+
+def test_check_crop_code_accepts_cas():
+    assert _check_crop_code("cas") == "cas"
+    with pytest.raises(ValueError):
+        _check_crop_code("unknown")

## Commit message
feat(crop): enable cassava ('cas') in ISIMIP→FAO workflow (minimal, no logic change)

- Add 'cas' to ISIMIP_CROP_CODES
- Map 'cas' → FAO "Cassava"; add pretty name
- Docs: list 'cas' in tutorial params
- Tests: mapping + validator
(No new deps; no behavior change)

## Notes
- References: limit to those strictly necessary in the PR description (e.g., ISIMIP crop code list for `cas`, FAO item name “Cassava”). Do not include extra nutritional references here.
