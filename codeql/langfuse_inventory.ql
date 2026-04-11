/**
 * Unified Langfuse/telemetry inventory.
 * Emits rows: file, line, function, kind, detail.
 * Uses modular TaintTracking for inter-procedural flow.
 */
import python
import semmle.python.dataflow.new.DataFlow
import semmle.python.dataflow.new.TaintTracking

predicate targetName(string n) {
  n in [
    "Langfuse",
    "LangfuseOpenAI",
    "propagate_attributes",
    "get_langfuse_client",
    "get_tracing_openai_client",
    "start_observation",
    "start_root_observation",
    "start_child_observation",
    "propagate_context",
    "flush_langfuse",
    "emit_score",
    "emit_score_for_handle",
    "_safe_end_observation",
    "_safe_end_span",
    "LocalizationEvaluationError"
  ]
}

predicate lifecycleName(string n) { n in ["end", "update", "score"] }

/** Recursively peel attributes to their base object. */
Expr attrBase(Expr e) {
  result = e
  or exists(Attribute a | result = attrBase(a.getObject()) and e = a)
}

predicate sourceHelperCall(Expr e) {
  e instanceof Call and
  (
    (e.(Call).getFunc() instanceof Name and
      e.(Call).getFunc().(Name).getId() in [
        "get_langfuse_client",
        "start_observation",
        "start_root_observation",
        "start_child_observation",
        "propagate_context",
        "emit_score",
        "emit_score_for_handle",
        "flush_langfuse"
      ])
    or
    (e.(Call).getFunc() instanceof Attribute and
      e.(Call).getFunc().(Attribute).getAttr() in [
        "get_langfuse_client",
        "start_observation",
        "start_root_observation",
        "start_child_observation",
        "propagate_context",
        "emit_score",
        "emit_score_for_handle",
        "flush_langfuse"
      ])
  )
}

predicate lifecycleSinkExpr(Expr sinkExpr) {
  exists(Call c2, Attribute a2 |
    c2.getFunc() = a2 and lifecycleName(a2.getAttr()) and
    sinkExpr = attrBase(a2.getObject())
  )
  or
  exists(Call c3 |
    sinkExpr = c3.getAnArg() and
    (
      (c3.getFunc() instanceof Name and c3.getFunc().(Name).getId() in ["emit_score", "emit_score_for_handle"]) or
      (c3.getFunc() instanceof Attribute and c3.getFunc().(Attribute).getAttr() in ["emit_score", "emit_score_for_handle"])
    )
  )
}

module LangfuseConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) { exists(Expr e | source = DataFlow::exprNode(e) and sourceHelperCall(e)) }
  predicate isSink(DataFlow::Node sink) { exists(Expr e | sink = DataFlow::exprNode(e) and lifecycleSinkExpr(e)) }
  predicate isBarrier(DataFlow::Node node) { none() }
  predicate isAdditionalFlowStep(DataFlow::Node nodeFrom, DataFlow::Node nodeTo) { none() }
}

module LangfuseFlow = TaintTracking::Global<LangfuseConfig>;

string scopeNameExpr(Expr e) {
  result = "<module>"
  or exists(Function f | e.getScope() = f and result = f.getName())
}

string scopeNameStmt(Stmt s) {
  result = "<module>"
  or exists(Function f | s.getScope() = f and result = f.getName())
}

predicate row(string file, int line, string func, string kind, string detail) {
  /* Imports */
  exists(Import imp, string mod |
    file = imp.getLocation().getFile().getRelativePath() and
    line = imp.getLocation().getStartLine() and
    func = scopeNameStmt(imp) and
    kind = "import" and
    mod = imp.getAnImportedModuleName() and
    mod.regexpMatch("^langfuse(\\.|$)") and
    detail = mod
  )
  or
  /* Names */
  exists(Name n |
    file = n.getLocation().getFile().getRelativePath() and
    line = n.getLocation().getStartLine() and
    func = scopeNameExpr(n) and
    kind = "name" and
    targetName(n.getId()) and
    detail = n.getId()
  )
  or
  /* Attributes */
  exists(Attribute a |
    file = a.getLocation().getFile().getRelativePath() and
    line = a.getLocation().getStartLine() and
    func = scopeNameExpr(a) and
    kind = "attr" and
    targetName(a.getAttr()) and
    detail = a.getAttr()
  )
  or
  /* Lifecycle calls */
  exists(Call c, Attribute a |
    c.getFunc() = a and lifecycleName(a.getAttr()) and
    file = a.getLocation().getFile().getRelativePath() and
    line = a.getLocation().getStartLine() and
    func = scopeNameExpr(a) and
    kind = "lifecycle-call" and
    detail = a.getAttr()
  )
  or
  /* Env strings */
  exists(StringLiteral s |
    s.getText().regexpMatch("LANGFUSE_") and
    file = s.getLocation().getFile().getRelativePath() and
    line = s.getLocation().getStartLine() and
    func = scopeNameExpr(s) and
    kind = "env-string" and
    detail = s.getText()
  )
  or
  /* Inter-procedural flow: sources -> lifecycle/score sinks */
  exists(Expr srcExpr, Expr sinkExpr |
    sourceHelperCall(srcExpr) and lifecycleSinkExpr(sinkExpr) and
    LangfuseFlow::flow(DataFlow::exprNode(srcExpr), DataFlow::exprNode(sinkExpr)) and
    file = srcExpr.getLocation().getFile().getRelativePath() and
    line = srcExpr.getLocation().getStartLine() and
    func = scopeNameExpr(srcExpr) and
    kind = "flow" and
    detail = sinkExpr.toString()
  )
}

from string file, int line, string func, string kind, string detail
where row(file, line, func, kind, detail)
select file, line, func, kind, detail
