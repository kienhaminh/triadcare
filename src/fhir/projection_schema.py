"""Allowlist schemas for the FHIR projection: which fields of each resource reach the model.

A leaf names the value type the projector accepts. TEXT is narrative that may hold XHTML;
the projector reduces it to plain text. An _array entry projects each item.
"""

from __future__ import annotations

from typing import Any

STRING = "string"
TEXT = "text"
BOOLEAN = "boolean"
NUMBER = "number"


def _array(item_schema: Any, *, deduplicate: bool = False) -> dict:
    return {"_array": item_schema, "_deduplicate": deduplicate}


CODING = {
    "system": STRING,
    "version": STRING,
    "code": STRING,
    "display": TEXT,
    "userSelected": BOOLEAN,
}
CODEABLE_CONCEPT = {
    "coding": _array(CODING, deduplicate=True),
    "text": TEXT,
}
REFERENCE = {"reference": STRING, "type": STRING, "display": TEXT}
QUANTITY = {
    "value": NUMBER,
    "comparator": STRING,
    "unit": STRING,
    "system": STRING,
    "code": STRING,
}
PERIOD = {"start": STRING, "end": STRING}
RANGE = {"low": QUANTITY, "high": QUANTITY}
RATIO = {"numerator": QUANTITY, "denominator": QUANTITY}
ANNOTATION = {
    "authorString": TEXT,
    "authorReference": REFERENCE,
    "time": STRING,
    "text": TEXT,
}
TIMING = {
    "event": _array(STRING),
    "repeat": {
        "boundsDuration": {"value": NUMBER, "unit": STRING, "system": STRING, "code": STRING},
        "boundsRange": RANGE,
        "boundsPeriod": PERIOD,
        "count": NUMBER,
        "countMax": NUMBER,
        "duration": NUMBER,
        "durationMax": NUMBER,
        "durationUnit": STRING,
        "frequency": NUMBER,
        "frequencyMax": NUMBER,
        "period": NUMBER,
        "periodMax": NUMBER,
        "periodUnit": STRING,
        "dayOfWeek": _array(STRING),
        "timeOfDay": _array(STRING),
        "when": _array(STRING),
        "offset": NUMBER,
    },
    "code": CODEABLE_CONCEPT,
}
DOSE_AND_RATE = {
    "type": CODEABLE_CONCEPT,
    "doseQuantity": QUANTITY,
    "doseRange": RANGE,
    "rateQuantity": QUANTITY,
    "rateRange": RANGE,
    "rateRatio": RATIO,
}
DOSAGE = {
    "sequence": NUMBER,
    "text": TEXT,
    "additionalInstruction": _array(CODEABLE_CONCEPT, deduplicate=True),
    "patientInstruction": TEXT,
    "timing": TIMING,
    "asNeededBoolean": BOOLEAN,
    "asNeededCodeableConcept": CODEABLE_CONCEPT,
    "site": CODEABLE_CONCEPT,
    "route": CODEABLE_CONCEPT,
    "method": CODEABLE_CONCEPT,
    "doseAndRate": _array(DOSE_AND_RATE),
    "maxDosePerPeriod": RATIO,
    "maxDosePerAdministration": QUANTITY,
    "maxDosePerLifetime": QUANTITY,
}
IDENTIFIER = {"system": STRING, "value": STRING}
ATTACHMENT_METADATA = {
    "contentType": STRING,
    "language": STRING,
    "title": TEXT,
    "creation": STRING,
    "size": NUMBER,
}

# lastUpdated is storage time, not a clinical date; agents reported it as one.
COMMON_META = {"versionId": STRING}
_NOTE = _array(ANNOTATION, deduplicate=True)

RESOURCE_SCHEMAS: dict[str, dict] = {
    "Patient": {
        "active": BOOLEAN,
        "name": _array({"use": STRING, "text": TEXT, "family": STRING, "given": _array(STRING)}),
        "gender": STRING,
        "birthDate": STRING,
        "deceasedBoolean": BOOLEAN,
        "deceasedDateTime": STRING,
        "communication": _array({"language": CODEABLE_CONCEPT, "preferred": BOOLEAN}),
        "generalPractitioner": _array(REFERENCE),
        "managingOrganization": REFERENCE,
    },
    "Condition": {
        "clinicalStatus": CODEABLE_CONCEPT,
        "verificationStatus": CODEABLE_CONCEPT,
        "category": _array(CODEABLE_CONCEPT, deduplicate=True),
        "severity": CODEABLE_CONCEPT,
        "code": CODEABLE_CONCEPT,
        "bodySite": _array(CODEABLE_CONCEPT, deduplicate=True),
        "encounter": REFERENCE,
        "onsetDateTime": STRING,
        "onsetAge": QUANTITY,
        "onsetPeriod": PERIOD,
        "onsetRange": RANGE,
        "onsetString": TEXT,
        "abatementBoolean": BOOLEAN,
        "abatementDateTime": STRING,
        "abatementAge": QUANTITY,
        "abatementPeriod": PERIOD,
        "abatementRange": RANGE,
        "abatementString": TEXT,
        "recordedDate": STRING,
        "stage": _array(
            {"summary": CODEABLE_CONCEPT, "assessment": _array(REFERENCE), "type": CODEABLE_CONCEPT}
        ),
        "evidence": _array(
            {"code": _array(CODEABLE_CONCEPT, deduplicate=True), "detail": _array(REFERENCE)}
        ),
        "note": _NOTE,
    },
    "Observation": {
        "status": STRING,
        "category": _array(CODEABLE_CONCEPT, deduplicate=True),
        "code": CODEABLE_CONCEPT,
        "effectiveDateTime": STRING,
        "effectivePeriod": PERIOD,
        "effectiveTiming": TIMING,
        "effectiveInstant": STRING,
        "issued": STRING,
        "performer": _array(REFERENCE),
        "valueQuantity": QUANTITY,
        "valueCodeableConcept": CODEABLE_CONCEPT,
        "valueString": TEXT,
        "valueBoolean": BOOLEAN,
        "valueInteger": NUMBER,
        "valueRange": RANGE,
        "valueRatio": RATIO,
        "valueTime": STRING,
        "valueDateTime": STRING,
        "valuePeriod": PERIOD,
        "dataAbsentReason": CODEABLE_CONCEPT,
        "interpretation": _array(CODEABLE_CONCEPT, deduplicate=True),
        "note": _NOTE,
        "bodySite": CODEABLE_CONCEPT,
        "method": CODEABLE_CONCEPT,
        "referenceRange": _array(
            {
                "low": QUANTITY,
                "high": QUANTITY,
                "type": CODEABLE_CONCEPT,
                "appliesTo": _array(CODEABLE_CONCEPT, deduplicate=True),
                "age": RANGE,
                "text": TEXT,
            }
        ),
        "hasMember": _array(REFERENCE),
        "derivedFrom": _array(REFERENCE),
        "component": _array(
            {
                "code": CODEABLE_CONCEPT,
                "valueQuantity": QUANTITY,
                "valueCodeableConcept": CODEABLE_CONCEPT,
                "valueString": TEXT,
                "valueBoolean": BOOLEAN,
                "valueInteger": NUMBER,
                "valueRange": RANGE,
                "valueRatio": RATIO,
                "valueTime": STRING,
                "valueDateTime": STRING,
                "valuePeriod": PERIOD,
                "dataAbsentReason": CODEABLE_CONCEPT,
                "interpretation": _array(CODEABLE_CONCEPT, deduplicate=True),
                "referenceRange": _array(
                    {"low": QUANTITY, "high": QUANTITY, "type": CODEABLE_CONCEPT, "text": TEXT}
                ),
            }
        ),
    },
    "MedicationStatement": {
        "status": STRING,
        "statusReason": _array(CODEABLE_CONCEPT, deduplicate=True),
        "medicationCodeableConcept": CODEABLE_CONCEPT,
        "medicationReference": REFERENCE,
        "effectiveDateTime": STRING,
        "effectivePeriod": PERIOD,
        "dateAsserted": STRING,
        "informationSource": REFERENCE,
        "derivedFrom": _array(REFERENCE),
        "partOf": _array(REFERENCE),
        "reasonCode": _array(CODEABLE_CONCEPT, deduplicate=True),
        "reasonReference": _array(REFERENCE),
        "dosage": _array(DOSAGE),
        "note": _NOTE,
        "taken": BOOLEAN,
    },
    "MedicationRequest": {
        "status": STRING,
        "statusReason": CODEABLE_CONCEPT,
        "statusChanged": STRING,
        "intent": STRING,
        "category": _array(CODEABLE_CONCEPT, deduplicate=True),
        "priority": STRING,
        "doNotPerform": BOOLEAN,
        "reportedBoolean": BOOLEAN,
        "reportedReference": REFERENCE,
        "medicationCodeableConcept": CODEABLE_CONCEPT,
        "medicationReference": REFERENCE,
        "encounter": REFERENCE,
        "supportingInformation": _array(REFERENCE),
        "authoredOn": STRING,
        "requester": REFERENCE,
        "performer": REFERENCE,
        "performerType": CODEABLE_CONCEPT,
        "reasonCode": _array(CODEABLE_CONCEPT, deduplicate=True),
        "reasonReference": _array(REFERENCE),
        "basedOn": _array(REFERENCE),
        "courseOfTherapyType": CODEABLE_CONCEPT,
        "dosageInstruction": _array(DOSAGE),
        "dispenseRequest": {
            "initialFill": {"quantity": QUANTITY, "duration": QUANTITY},
            "dispenseInterval": QUANTITY,
            "validityPeriod": PERIOD,
            "numberOfRepeatsAllowed": NUMBER,
            "quantity": QUANTITY,
            "expectedSupplyDuration": QUANTITY,
            "performer": REFERENCE,
        },
        "substitution": {
            "allowedBoolean": BOOLEAN,
            "allowedCodeableConcept": CODEABLE_CONCEPT,
            "reason": CODEABLE_CONCEPT,
        },
        "priorPrescription": REFERENCE,
        "note": _NOTE,
    },
    "AllergyIntolerance": {
        "clinicalStatus": CODEABLE_CONCEPT,
        "verificationStatus": CODEABLE_CONCEPT,
        "type": STRING,
        "category": _array(STRING),
        "criticality": STRING,
        "code": CODEABLE_CONCEPT,
        "encounter": REFERENCE,
        "onsetDateTime": STRING,
        "onsetAge": QUANTITY,
        "onsetPeriod": PERIOD,
        "onsetRange": RANGE,
        "onsetString": TEXT,
        "recordedDate": STRING,
        "participant": _array({"function": CODEABLE_CONCEPT, "actor": REFERENCE}),
        "lastOccurrence": STRING,
        "note": _NOTE,
        "reaction": _array(
            {
                "substance": CODEABLE_CONCEPT,
                "manifestation": _array(CODEABLE_CONCEPT, deduplicate=True),
                "description": TEXT,
                "onset": STRING,
                "severity": STRING,
                "exposureRoute": CODEABLE_CONCEPT,
                "note": _NOTE,
            }
        ),
    },
    "Encounter": {
        "status": STRING,
        "statusHistory": _array({"status": STRING, "period": PERIOD}),
        "class": CODING,
        "classHistory": _array({"class": CODING, "period": PERIOD}),
        "type": _array(CODEABLE_CONCEPT, deduplicate=True),
        "serviceType": CODEABLE_CONCEPT,
        "priority": CODEABLE_CONCEPT,
        "episodeOfCare": _array(REFERENCE),
        "basedOn": _array(REFERENCE),
        "participant": _array(
            {
                "type": _array(CODEABLE_CONCEPT, deduplicate=True),
                "period": PERIOD,
                "individual": REFERENCE,
            }
        ),
        "appointment": _array(REFERENCE),
        "period": PERIOD,
        "length": QUANTITY,
        "reasonCode": _array(CODEABLE_CONCEPT, deduplicate=True),
        "reasonReference": _array(REFERENCE),
        "diagnosis": _array({"condition": REFERENCE, "use": CODEABLE_CONCEPT, "rank": NUMBER}),
        "hospitalization": {
            "origin": REFERENCE,
            "admitSource": CODEABLE_CONCEPT,
            "reAdmission": CODEABLE_CONCEPT,
            "dietPreference": _array(CODEABLE_CONCEPT, deduplicate=True),
            "specialCourtesy": _array(CODEABLE_CONCEPT, deduplicate=True),
            "specialArrangement": _array(CODEABLE_CONCEPT, deduplicate=True),
            "destination": REFERENCE,
            "dischargeDisposition": CODEABLE_CONCEPT,
        },
        "location": _array(
            {
                "location": REFERENCE,
                "status": STRING,
                "physicalType": CODEABLE_CONCEPT,
                "period": PERIOD,
            }
        ),
        "serviceProvider": REFERENCE,
        "partOf": REFERENCE,
        "note": _NOTE,
    },
    "Procedure": {
        "status": STRING,
        "statusReason": CODEABLE_CONCEPT,
        "category": CODEABLE_CONCEPT,
        "code": CODEABLE_CONCEPT,
        "doNotPerform": BOOLEAN,
        "encounter": REFERENCE,
        "performedDateTime": STRING,
        "performedPeriod": PERIOD,
        "performedString": TEXT,
        "performedAge": QUANTITY,
        "performedRange": RANGE,
        "recorder": REFERENCE,
        "reportedBoolean": BOOLEAN,
        "reportedReference": REFERENCE,
        "performer": _array(
            {"function": CODEABLE_CONCEPT, "actor": REFERENCE, "onBehalfOf": REFERENCE}
        ),
        "location": REFERENCE,
        "reasonCode": _array(CODEABLE_CONCEPT, deduplicate=True),
        "reasonReference": _array(REFERENCE),
        "bodySite": _array(CODEABLE_CONCEPT, deduplicate=True),
        "outcome": CODEABLE_CONCEPT,
        "report": _array(REFERENCE),
        "complication": _array(CODEABLE_CONCEPT, deduplicate=True),
        "complicationDetail": _array(REFERENCE),
        "followUp": _array(CODEABLE_CONCEPT, deduplicate=True),
        "usedReference": _array(REFERENCE),
        "usedCode": _array(CODEABLE_CONCEPT, deduplicate=True),
        "note": _NOTE,
    },
    "DiagnosticReport": {
        "status": STRING,
        "category": _array(CODEABLE_CONCEPT, deduplicate=True),
        "code": CODEABLE_CONCEPT,
        "encounter": REFERENCE,
        "effectiveDateTime": STRING,
        "effectivePeriod": PERIOD,
        "issued": STRING,
        "performer": _array(REFERENCE),
        "resultsInterpreter": _array(REFERENCE),
        "specimen": _array(REFERENCE),
        "result": _array(REFERENCE),
        "note": _NOTE,
        "conclusion": TEXT,
        "conclusionCode": _array(CODEABLE_CONCEPT, deduplicate=True),
        "presentedForm": _array(ATTACHMENT_METADATA),
    },
    "CarePlan": {
        "status": STRING,
        "intent": STRING,
        "category": _array(CODEABLE_CONCEPT, deduplicate=True),
        "title": TEXT,
        "description": TEXT,
        "supportingInfo": _array(REFERENCE),
        "goal": _array(REFERENCE),
        "activity": _array(
            {
                "outcomeCodeableConcept": _array(CODEABLE_CONCEPT, deduplicate=True),
                "outcomeReference": _array(REFERENCE),
                "progress": _NOTE,
                "reference": REFERENCE,
                "detail": {
                    "kind": STRING,
                    "code": CODEABLE_CONCEPT,
                    "reasonCode": _array(CODEABLE_CONCEPT, deduplicate=True),
                    "reasonReference": _array(REFERENCE),
                    "goal": _array(REFERENCE),
                    "status": STRING,
                    "statusReason": CODEABLE_CONCEPT,
                    "doNotPerform": BOOLEAN,
                    "scheduledTiming": TIMING,
                    "scheduledPeriod": PERIOD,
                    "scheduledString": TEXT,
                    "location": REFERENCE,
                    "performer": _array(REFERENCE),
                    "productCodeableConcept": CODEABLE_CONCEPT,
                    "productReference": REFERENCE,
                    "dailyAmount": QUANTITY,
                    "quantity": QUANTITY,
                    "description": TEXT,
                },
            }
        ),
        "period": PERIOD,
        "created": STRING,
        "note": _NOTE,
    },
    "ImagingStudy": {
        # The identifier is the study ID of the imaging catalog; the description states
        # what the record does and does not establish.
        "identifier": _array(IDENTIFIER),
        "status": STRING,
        "description": TEXT,
        "modality": _array(CODING, deduplicate=True),
        "started": STRING,
        "numberOfSeries": NUMBER,
        "numberOfInstances": NUMBER,
        "procedureCode": _array(CODEABLE_CONCEPT, deduplicate=True),
        "reasonCode": _array(CODEABLE_CONCEPT, deduplicate=True),
        "reasonReference": _array(REFERENCE),
        "note": _NOTE,
        "series": _array(
            {
                "number": NUMBER,
                "modality": CODING,
                "description": TEXT,
                "numberOfInstances": NUMBER,
                "bodySite": CODING,
                "laterality": CODING,
                "started": STRING,
                "specimen": _array(REFERENCE),
            }
        ),
    },
    "DocumentReference": {
        "status": STRING,
        "docStatus": STRING,
        "type": CODEABLE_CONCEPT,
        "category": _array(CODEABLE_CONCEPT, deduplicate=True),
        "date": STRING,
        "author": _array(REFERENCE),
        "authenticator": REFERENCE,
        "custodian": REFERENCE,
        "relatesTo": _array({"code": STRING, "target": REFERENCE}),
        "description": TEXT,
        "securityLabel": _array(CODEABLE_CONCEPT, deduplicate=True),
        "content": _array(
            {
                "attachment": ATTACHMENT_METADATA,
                "format": CODING,
            }
        ),
        "context": {
            "encounter": _array(REFERENCE),
            "event": _array(CODEABLE_CONCEPT, deduplicate=True),
            "period": PERIOD,
            "facilityType": CODEABLE_CONCEPT,
            "practiceSetting": CODEABLE_CONCEPT,
            "sourcePatientInfo": REFERENCE,
            "related": _array(REFERENCE),
        },
    },
}
