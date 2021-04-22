import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  ElementRef,
  Inject,
  Input,
  OnInit
} from "@angular/core";
import { fromEvent } from "rxjs";
import { WINDOW } from "@ng-web-apis/common";
import { map, startWith, takeUntil } from "rxjs/operators";
import { animate, style, transition, trigger } from "@angular/animations";
import { round, TuiDestroyService } from '@taiga-ui/cdk';

/**
 * Define ID type
 */
export type ID = string;

/**
 * Node of the graph
 */
export interface Node {
  id: ID;
  label: string;
  labelPosition?: "center" | "above" | "below";
}

export interface Relevance {
  pred: string;
  relevance: number;
}

/**
 * Edge of the graph
 */
export interface Edge {
  id: ID;
  label?: string;
  top_10: Relevance[];
  start: ID;
  end: ID;
}

/**
 * Graph
 */
export interface Graph {
  nodes: Node[];
  edges: Edge[];
}

/**
 * Node position
 */
interface NodePosition {
  id: ID;
  label: string;
  hovered: boolean;
  cx: number;
  cy: number;
}

/**
 * Edge position
 */
interface EdgePosition {
  id: ID;
  label: string;
  top_10: Relevance[];
  backward: boolean;
  cx1: number;
  cy1: number;
  cx2: number;
  cy2: number;
}

/**
 * A component to display a graph.
 */
@Component({
  selector: "app-graph",
  templateUrl: "./graph.component.html",
  styleUrls: ["./graph.component.scss"],
  changeDetection: ChangeDetectionStrategy.OnPush,
  animations: [
    trigger("posAnimation", [
      transition(":enter", [
        style({ r: 0 }),
        animate("300ms ease-out", style({ r: "*" }))
      ])
    ])
  ],
  providers: [TuiDestroyService]
})
export class GraphComponent implements OnInit {
  // input graph
  @Input() graph: Graph;
  // input node radius
  @Input() nodeRadius: number = 20;

  open = false;

  // computed node positions
  nodePositions: NodePosition[] = [];
  // computed edge positions
  edgePositions: EdgePosition[] = [];

  // number of nodes given as input
  private _numberOfNodes: number;

  // emits resize events
  private readonly _resize$ = fromEvent(this.windowRef, "resize").pipe(
    map(event => (event.target as Window).innerWidth),
    startWith(this.windowRef.innerWidth)
  );

  constructor(
    private _el: ElementRef,
    private _cdr: ChangeDetectorRef,
    @Inject(TuiDestroyService) private readonly destroy$: TuiDestroyService,
    @Inject(WINDOW) readonly windowRef: Window
  ) { }

  ngOnInit() {
    this._numberOfNodes = this.graph.nodes.length;

    // reorder nodes if necessary
    let nodeEnd: ID;
    if (this._numberOfNodes > 2) {
      nodeEnd = this._getCenterNode("end");

      if (!nodeEnd) {
        nodeEnd = this._getCenterNode("start");
      }

      if (nodeEnd) {
        this._reorderCenterNode(nodeEnd);
      }
    }

    // compute coordinates of nodes and edges on resize and initial loading
    this._resize$.pipe(takeUntil(this.destroy$)).subscribe(() => {
      this.nodePositions = this._computeNodesCoordinates();
      this.edgePositions = this._computeEdgesCoordinates();
      this._cdr.markForCheck();
    });
  }

  /**
   * Compute nodes coordinates
   */
  private _computeNodesCoordinates(): NodePosition[] {
    const nodePositions: NodePosition[] = [];

    const hostHeight = this._el.nativeElement.offsetHeight;
    const hostWidth = this._el.nativeElement.offsetWidth;

    const partitionWidth = hostWidth / this._numberOfNodes;

    const cx = partitionWidth * 0.5;
    const cy = hostHeight * 0.5;

    this.graph.nodes.forEach((node, i) =>
      nodePositions.push({
        id: node.id,
        label: node.label,
        hovered: false,
        cx: cx + partitionWidth * i,
        cy: i % 2 === 0 ? cy * 0.7 : cy * 1.2
      })
    );

    return nodePositions;
  }

  /**
   * Compute edges coordinates
   */
  private _computeEdgesCoordinates(): EdgePosition[] {
    const edgePositions: EdgePosition[] = [];

    this.graph.edges.forEach(edge => {
      const positions1 = this.nodePositions.find(
        positions => edge.start === positions.id
      );
      const positions2 = this.nodePositions.find(
        positions => edge.end === positions.id
      );

      const isBackward = positions1.cx > positions2.cx;

      const [cx1, cx2] = this._computeEdgeX(
        positions1.cx,
        positions2.cx,
        isBackward
      );
      const [cy1, cy2] = this._computeEdgeY(
        positions1.cy,
        positions2.cy,
        isBackward
      );

      edgePositions.push({
        id: edge.id,
        label: edge.label,
        top_10: edge.top_10,
        backward: isBackward,
        cx1,
        cy1,
        cx2,
        cy2
      });
    });

    return edgePositions;
  }

  /**
   * Computes edge x coordinate
   */
  private _computeEdgeX(
    cx1: number,
    cx2: number,
    isBackward: boolean
  ): number[] {
    if (isBackward) {
      return [cx2 + this.nodeRadius, cx1 - this.nodeRadius];
    }
    return [cx1 + this.nodeRadius, cx2 - this.nodeRadius];
  }

  /**
   * Computes edge y coordinate
   */
  private _computeEdgeY(
    cy1: number,
    cy2: number,
    isBackward: boolean
  ): number[] {
    if (isBackward) {
      return [cy2, cy1];
    }
    return [cy1, cy2];
  }

  /**
   * Get node with in-degree or out-degree which is more than 1
   */
  private _getCenterNode(source: "start" | "end"): ID {
    for (const edge of this.graph.edges) {
      const edgeTmp = this.graph.edges.find(
        e => e.id !== edge.id && e[source] === edge[source]
      );
      if (edgeTmp) {
        return edgeTmp[source];
      }
    }
    return null;
  }

  /**
   * Reorder center node so that it's the central node
   */
  private _reorderCenterNode(nodeID: ID): void {
    const itemIndex = this.graph.nodes.findIndex(node => node.id === nodeID);
    const item = this.graph.nodes[itemIndex];
    if (itemIndex !== 1) {
      this.graph.nodes.splice(itemIndex, 1);
      this.graph.nodes.splice(1, 0, item);
    }
  }

  trackNodes(index, node): any {
    return node.id;
  }

  trackEdges(index, edge): any {
    return edge.id;
  }

  getLabel(fullText: string): string {
    return fullText.split('/').pop().split('>')[0];
  }

  getLink(resource: string): string {
    return resource.split('<')[1].split('>')[0];
  }

  getRelevance(relevance: number): number {
    return round(relevance, 3)
  }

  onEdgeHover(): void {
    this.open = true;
  }

  onEdgeLeave(): void {
    this.open = false;
  }
}
